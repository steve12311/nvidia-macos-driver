use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{Capability, Decoration, MemoryAccess, Op, StorageClass};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Caps {
    pub int8: bool,
    pub int16: bool,
    pub float16: bool,
}

impl Default for Caps {
    fn default() -> Self {
        Self { int8: true, int16: true, float16: true }
    }
}

fn id(o: &Operand) -> Option<u32> {
    if let Operand::IdRef(v) = o { Some(*v) } else { None }
}

fn lit32(o: &Operand) -> Option<u32> {
    if let Operand::LiteralBit32(v) = o { Some(*v) } else { None }
}

fn find_or_push_type(
    types: &mut Vec<Instruction>,
    op: Op,
    operands: Vec<Operand>,
    fresh: &mut impl FnMut() -> u32,
) -> u32 {
    if let Some(n) = types
        .iter()
        .find(|t| t.class.opcode == op && t.operands == operands)
        .and_then(|t| t.result_id)
    {
        return n;
    }
    let n = fresh();
    types.push(Instruction::new(op, None, Some(n), operands));
    n
}

fn find_or_push_const_u64(
    types: &mut Vec<Instruction>,
    ulong: u32,
    val: u64,
    fresh: &mut impl FnMut() -> u32,
) -> u32 {
    if let Some(n) = types
        .iter()
        .find(|i| {
            i.class.opcode == Op::Constant
                && i.result_type == Some(ulong)
                && i.operands == [Operand::LiteralBit64(val)]
        })
        .and_then(|i| i.result_id)
    {
        return n;
    }
    let n = fresh();
    types.push(Instruction::new(
        Op::Constant,
        Some(ulong),
        Some(n),
        vec![Operand::LiteralBit64(val)],
    ));
    n
}

enum Route {
    /// Dynamic subword lane from a proven byte index into a PhysicalStorageBuffer
    /// word array whose block member 0 starts at offset 0.
    DynamicByte {
        root: u32,
        zero_const: u32,
        ptr_uint_ty: u32,
        byte_idx: u32,
    },
    /// Static subword lane `byte_delta` bytes inside a PhysicalStorageBuffer word pointer.
    StaticLane {
        word_ptr: u32,
        byte_delta: u64,
    },
}

struct Rewrite {
    route: Route,
    elem_ty: u32,
    elem_size: u32,
    res_id: u32,
    extra_flags: MemoryAccess,
    /// Trailing operands after the Aligned literal (e.g. scope IDs for MakePointerVisible).
    trailing_ops: Vec<Operand>,
}

fn parse_load_access(load: &Instruction) -> Option<(MemoryAccess, Vec<Operand>)> {
    match load.operands.get(1) {
        None => Some((MemoryAccess::NONE, Vec::new())),
        Some(Operand::MemoryAccess(flags)) => {
            if flags.contains(MemoryAccess::VOLATILE) {
                return None;
            }
            let skip = if flags.contains(MemoryAccess::ALIGNED) { 3 } else { 2 };
            let trailing = load.operands.get(skip..).unwrap_or(&[]).to_vec();
            Some((*flags & !MemoryAccess::ALIGNED, trailing))
        }
        _ => None,
    }
}

fn match_bitand_const(
    inst: &Instruction,
    uint: u32,
    consts_u32: &HashMap<u32, u32>,
    target: u32,
) -> Option<u32> {
    if inst.class.opcode != Op::BitwiseAnd || inst.result_type != Some(uint) {
        return None;
    }
    match inst.operands.as_slice() {
        [Operand::IdRef(a), Operand::IdRef(b)] => {
            if consts_u32.get(b) == Some(&target) {
                Some(*a)
            } else if consts_u32.get(a) == Some(&target) {
                Some(*b)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Lowers 32-bit word-backed 8-bit/16-bit subword extractions on `PhysicalStorageBuffer`
/// pointers directly to native 8-bit/16-bit `PhysicalStorageBuffer` `OpLoad` instructions.
///
/// Regular descriptor buffers (`StorageClass::StorageBuffer`, including `kept_roots`) are
/// never touched, preserving their existing `ArrayStride 4` descriptor layout compatibility.
/// Native 8-bit/16-bit loads are emitted only when both the module's SPIR-V capabilities
/// (`Int8`, `Int16`, `Float16`) and the runtime device capabilities (`caps`) permit them.
pub fn lower(m: &mut Module, next: &mut u32, uint: u32, ulong: u32, caps: Caps) -> usize {
    let has_cap = |cap: Capability| {
        m.capabilities
            .iter()
            .any(|i| i.operands == [Operand::Capability(cap)])
    };
    let allow_i8 = caps.int8 && has_cap(Capability::Int8);
    let allow_i16 = caps.int16 && has_cap(Capability::Int16);
    let allow_f16 = caps.float16 && has_cap(Capability::Float16);
    if !allow_i8 && !allow_i16 && !allow_f16 {
        return 0;
    }

    let defs: HashMap<u32, Instruction> = m
        .types_global_values
        .iter()
        .filter_map(|i| Some((i.result_id?, i.clone())))
        .collect();
    let consts_u32: HashMap<u32, u32> = m
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Constant && i.result_type == Some(uint))
        .filter_map(|i| Some((i.result_id?, lit32(i.operands.first()?)?)))
        .collect();

    // Only PhysicalStorageBuffer pointers to `uint` qualify; StorageBuffer descriptors stay untouched.
    let psb_uint_ptrs: HashSet<u32> = defs
        .iter()
        .filter_map(|(&n, i)| {
            (i.class.opcode == Op::TypePointer
                && i.operands
                    == [
                        Operand::StorageClass(StorageClass::PhysicalStorageBuffer),
                        Operand::IdRef(uint),
                    ])
            .then_some(n)
        })
        .collect();
    if psb_uint_ptrs.is_empty() {
        return 0;
    }

    // Track PhysicalStorageBuffer block pointer types whose struct member 0 is at byte offset 0
    // with an ArrayStride-4 uint runtime array.
    let zero_offset_blocks: HashSet<u32> = m
        .annotations
        .iter()
        .filter_map(|a| match a.operands.as_slice() {
            [Operand::IdRef(s), Operand::LiteralBit32(0), Operand::Decoration(Decoration::Offset), Operand::LiteralBit32(0)] => {
                Some(*s)
            }
            _ => None,
        })
        .collect();
    let stride4_arrays: HashSet<u32> = m
        .annotations
        .iter()
        .filter_map(|a| match a.operands.as_slice() {
            [Operand::IdRef(arr), Operand::Decoration(Decoration::ArrayStride), Operand::LiteralBit32(4)] => {
                Some(*arr)
            }
            _ => None,
        })
        .collect();
    let psb_word_block_ptrs: HashSet<u32> = defs
        .iter()
        .filter_map(|(&n, i)| {
            if i.class.opcode != Op::TypePointer
                || i.operands.first() != Some(&Operand::StorageClass(StorageClass::PhysicalStorageBuffer))
            {
                return None;
            }
            let block_ty = id(i.operands.get(1)?)?;
            if !zero_offset_blocks.contains(&block_ty) {
                return None;
            }
            let block = defs.get(&block_ty)?;
            if block.class.opcode != Op::TypeStruct {
                return None;
            }
            let arr_ty = id(block.operands.first()?)?;
            if !stride4_arrays.contains(&arr_ty) {
                return None;
            }
            let arr = defs.get(&arr_ty)?;
            if !matches!(arr.class.opcode, Op::TypeRuntimeArray | Op::TypeArray)
                || arr.operands.first() != Some(&Operand::IdRef(uint))
            {
                return None;
            }
            Some(n)
        })
        .collect();

    let subword_vec_elem = |vec_ty: u32| -> Option<(u32, u32, u32)> {
        let v = defs.get(&vec_ty)?;
        if v.class.opcode != Op::TypeVector {
            return None;
        }
        let elem_ty = id(v.operands.first()?)?;
        let lanes = lit32(v.operands.get(1)?)?;
        let elem = defs.get(&elem_ty)?;
        match (elem.class.opcode, elem.operands.first().and_then(lit32), lanes) {
            (Op::TypeInt, Some(8), 4) if allow_i8 => Some((elem_ty, 1, 4)),
            (Op::TypeInt, Some(16), 2) if allow_i16 => Some((elem_ty, 2, 2)),
            (Op::TypeFloat, Some(16), 2) if allow_f16 => Some((elem_ty, 2, 2)),
            _ => None,
        }
    };

    let mut fresh = || {
        let n = *next;
        *next += 1;
        n
    };
    let mut psb_elem_ptrs: HashMap<u32, u32> = HashMap::new();
    let mut lowered_total = 0usize;

    for fi in 0..m.functions.len() {
        let mut local: HashMap<u32, Instruction> = HashMap::new();
        let mut uses: HashMap<u32, usize> = HashMap::new();
        let mut def_loc: HashMap<u32, (usize, usize)> = HashMap::new();
        let mut value_types: HashMap<u32, u32> = HashMap::new();
        for p in &m.functions[fi].parameters {
            if let (Some(r), Some(t)) = (p.result_id, p.result_type) {
                value_types.insert(r, t);
            }
        }
        for (bi, blk) in m.functions[fi].blocks.iter().enumerate() {
            for (ii, inst) in blk.instructions.iter().enumerate() {
                if let Some(r) = inst.result_id {
                    local.insert(r, inst.clone());
                    def_loc.insert(r, (bi, ii));
                    if let Some(t) = inst.result_type {
                        value_types.insert(r, t);
                    }
                }
                for o in &inst.operands {
                    if let Operand::IdRef(v) = o {
                        *uses.entry(*v).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut dce_candidates: HashSet<u32> = HashSet::new();
        let mut fn_changed = false;

        for bi in 0..m.functions[fi].blocks.len() {
            let body = &m.functions[fi].blocks[bi].instructions;
            let mut rewrites: HashMap<usize, Rewrite> = HashMap::new();
            let mut remove_idx: HashSet<usize> = HashSet::new();

            for (ei, inst) in body.iter().enumerate() {
                let (Some(res_id), Some(res_ty)) = (inst.result_id, inst.result_type) else {
                    continue;
                };
                match inst.class.opcode {
                    Op::VectorExtractDynamic => {
                        let [Operand::IdRef(vec_id), Operand::IdRef(lane_id)] =
                            inst.operands.as_slice()
                        else {
                            continue;
                        };
                        if uses.get(vec_id) != Some(&1) {
                            continue;
                        }
                        let Some(&(vec_bi, vec_ii)) = def_loc.get(vec_id) else {
                            continue;
                        };
                        if vec_bi != bi {
                            continue;
                        }
                        let vec_def = &local[vec_id];
                        if vec_def.class.opcode != Op::Bitcast {
                            continue;
                        }
                        let Some(vec_ty) = vec_def.result_type else {
                            continue;
                        };
                        let Some((elem_ty, elem_size, _)) = subword_vec_elem(vec_ty) else {
                            continue;
                        };
                        if elem_ty != res_ty {
                            continue;
                        }
                        let [Operand::IdRef(word_id)] = vec_def.operands.as_slice() else {
                            continue;
                        };
                        if uses.get(word_id) != Some(&1) {
                            continue;
                        }
                        let Some(&(word_bi, word_ii)) = def_loc.get(word_id) else {
                            continue;
                        };
                        if word_bi != bi || rewrites.contains_key(&word_ii) {
                            continue;
                        }
                        let word_def = &local[word_id];
                        if word_def.class.opcode != Op::Load || word_def.result_type != Some(uint) {
                            continue;
                        }
                        let Some((extra_flags, trailing_ops)) = parse_load_access(word_def) else {
                            continue;
                        };
                        let Some(p_id) = word_def.operands.first().and_then(id) else {
                            continue;
                        };
                        let Some(p_def) = local.get(&p_id) else {
                            continue;
                        };
                        let Some(ptr_uint_ty) = p_def.result_type else {
                            continue;
                        };
                        if !psb_uint_ptrs.contains(&ptr_uint_ty)
                            || !matches!(
                                p_def.class.opcode,
                                Op::AccessChain | Op::InBoundsAccessChain
                            )
                        {
                            continue;
                        }
                        let [Operand::IdRef(root), Operand::IdRef(m0), Operand::IdRef(word_idx)] =
                            p_def.operands.as_slice()
                        else {
                            continue;
                        };
                        if consts_u32.get(m0) != Some(&0) {
                            continue;
                        }
                        if !value_types
                            .get(root)
                            .is_some_and(|t| psb_word_block_ptrs.contains(t))
                        {
                            continue;
                        }
                        let Some(w_def) = local.get(word_idx) else {
                            continue;
                        };
                        if w_def.class.opcode != Op::UDiv || w_def.result_type != Some(uint) {
                            continue;
                        }
                        let [Operand::IdRef(byte_idx), Operand::IdRef(c4)] =
                            w_def.operands.as_slice()
                        else {
                            continue;
                        };
                        if consts_u32.get(c4) != Some(&4) {
                            continue;
                        }
                        let Some(lane_def) = local.get(lane_id) else {
                            continue;
                        };
                        let lane_ok = match elem_size {
                            1 => {
                                match_bitand_const(lane_def, uint, &consts_u32, 3) == Some(*byte_idx)
                            }
                            2 => {
                                if lane_def.class.opcode == Op::ShiftRightLogical
                                    && lane_def.result_type == Some(uint)
                                {
                                    match lane_def.operands.as_slice() {
                                        [Operand::IdRef(masked), Operand::IdRef(c1)]
                                            if consts_u32.get(c1) == Some(&1) =>
                                        {
                                            let matched = local.get(masked).is_some_and(|md| {
                                                match_bitand_const(md, uint, &consts_u32, 3)
                                                    == Some(*byte_idx)
                                            });
                                            if matched {
                                                dce_candidates.insert(*masked);
                                            }
                                            matched
                                        }
                                        _ => false,
                                    }
                                } else {
                                    false
                                }
                            }
                            _ => false,
                        };
                        if !lane_ok {
                            continue;
                        }

                        dce_candidates.extend([p_id, *word_idx, *lane_id]);
                        remove_idx.insert(vec_ii);
                        remove_idx.insert(ei);
                        rewrites.insert(
                            word_ii,
                            Rewrite {
                                route: Route::DynamicByte {
                                    root: *root,
                                    zero_const: *m0,
                                    ptr_uint_ty,
                                    byte_idx: *byte_idx,
                                },
                                elem_ty,
                                elem_size,
                                res_id,
                                extra_flags,
                                trailing_ops,
                            },
                        );
                    }
                    Op::CompositeExtract => {
                        let [Operand::IdRef(vec_id), Operand::LiteralBit32(lane)] =
                            inst.operands.as_slice()
                        else {
                            continue;
                        };
                        if uses.get(vec_id) != Some(&1) {
                            continue;
                        }
                        let Some(&(vec_bi, vec_ii)) = def_loc.get(vec_id) else {
                            continue;
                        };
                        if vec_bi != bi {
                            continue;
                        }
                        let vec_def = &local[vec_id];
                        if vec_def.class.opcode != Op::Bitcast {
                            continue;
                        }
                        let Some(vec_ty) = vec_def.result_type else {
                            continue;
                        };
                        let Some((elem_ty, elem_size, lanes)) = subword_vec_elem(vec_ty) else {
                            continue;
                        };
                        if elem_ty != res_ty || *lane >= lanes {
                            continue;
                        }
                        let [Operand::IdRef(word_id)] = vec_def.operands.as_slice() else {
                            continue;
                        };
                        if uses.get(word_id) != Some(&1) {
                            continue;
                        }
                        let Some(&(word_bi, word_ii)) = def_loc.get(word_id) else {
                            continue;
                        };
                        if word_bi != bi || rewrites.contains_key(&word_ii) {
                            continue;
                        }
                        let word_def = &local[word_id];
                        if word_def.class.opcode != Op::Load || word_def.result_type != Some(uint) {
                            continue;
                        }
                        let Some((extra_flags, trailing_ops)) = parse_load_access(word_def) else {
                            continue;
                        };
                        let Some(p_id) = word_def.operands.first().and_then(id) else {
                            continue;
                        };
                        if !value_types
                            .get(&p_id)
                            .is_some_and(|t| psb_uint_ptrs.contains(t))
                        {
                            continue;
                        }

                        remove_idx.insert(vec_ii);
                        remove_idx.insert(ei);
                        rewrites.insert(
                            word_ii,
                            Rewrite {
                                route: Route::StaticLane {
                                    word_ptr: p_id,
                                    byte_delta: u64::from(*lane * elem_size),
                                },
                                elem_ty,
                                elem_size,
                                res_id,
                                extra_flags,
                                trailing_ops,
                            },
                        );
                    }
                    _ => {}
                }
            }

            if rewrites.is_empty() {
                continue;
            }
            fn_changed = true;
            let body = std::mem::take(&mut m.functions[fi].blocks[bi].instructions);
            let mut out = Vec::with_capacity(body.len() + 5 * rewrites.len());
            for (ii, inst) in body.into_iter().enumerate() {
                if remove_idx.contains(&ii) {
                    continue;
                }
                let Some(rw) = rewrites.remove(&ii) else {
                    out.push(inst);
                    continue;
                };
                let psb_elem_ptr_ty = *psb_elem_ptrs.entry(rw.elem_ty).or_insert_with(|| {
                    find_or_push_type(
                        &mut m.types_global_values,
                        Op::TypePointer,
                        vec![
                            Operand::StorageClass(StorageClass::PhysicalStorageBuffer),
                            Operand::IdRef(rw.elem_ty),
                        ],
                        &mut fresh,
                    )
                });
                let addr = match rw.route {
                    Route::DynamicByte {
                        root,
                        zero_const,
                        ptr_uint_ty,
                        byte_idx,
                    } => {
                        let (p0, base, off, a) = (fresh(), fresh(), fresh(), fresh());
                        out.push(Instruction::new(
                            Op::InBoundsAccessChain,
                            Some(ptr_uint_ty),
                            Some(p0),
                            vec![
                                Operand::IdRef(root),
                                Operand::IdRef(zero_const),
                                Operand::IdRef(zero_const),
                            ],
                        ));
                        out.push(Instruction::new(
                            Op::ConvertPtrToU,
                            Some(ulong),
                            Some(base),
                            vec![Operand::IdRef(p0)],
                        ));
                        out.push(Instruction::new(
                            Op::UConvert,
                            Some(ulong),
                            Some(off),
                            vec![Operand::IdRef(byte_idx)],
                        ));
                        out.push(Instruction::new(
                            Op::IAdd,
                            Some(ulong),
                            Some(a),
                            vec![Operand::IdRef(base), Operand::IdRef(off)],
                        ));
                        a
                    }
                    Route::StaticLane {
                        word_ptr,
                        byte_delta,
                    } => {
                        let base = fresh();
                        out.push(Instruction::new(
                            Op::ConvertPtrToU,
                            Some(ulong),
                            Some(base),
                            vec![Operand::IdRef(word_ptr)],
                        ));
                        if byte_delta == 0 {
                            base
                        } else {
                            let delta_c = find_or_push_const_u64(
                                &mut m.types_global_values,
                                ulong,
                                byte_delta,
                                &mut fresh,
                            );
                            let a = fresh();
                            out.push(Instruction::new(
                                Op::IAdd,
                                Some(ulong),
                                Some(a),
                                vec![Operand::IdRef(base), Operand::IdRef(delta_c)],
                            ));
                            a
                        }
                    }
                };
                let elem_ptr = fresh();
                out.push(Instruction::new(
                    Op::ConvertUToPtr,
                    Some(psb_elem_ptr_ty),
                    Some(elem_ptr),
                    vec![Operand::IdRef(addr)],
                ));
                let mut load_ops = vec![
                    Operand::IdRef(elem_ptr),
                    Operand::MemoryAccess(rw.extra_flags | MemoryAccess::ALIGNED),
                    Operand::LiteralBit32(rw.elem_size),
                ];
                load_ops.extend(rw.trailing_ops);
                out.push(Instruction::new(
                    Op::Load,
                    Some(rw.elem_ty),
                    Some(rw.res_id),
                    load_ops,
                ));
                lowered_total += 1;
            }
            m.functions[fi].blocks[bi].instructions = out;
        }

        // Remove only now-dead pure helper instructions that belonged to matched dynamic patterns.
        if fn_changed && !dce_candidates.is_empty() {
            loop {
                let mut live_uses: HashMap<u32, usize> = HashMap::new();
                for blk in &m.functions[fi].blocks {
                    for inst in &blk.instructions {
                        for o in &inst.operands {
                            if let Operand::IdRef(v) = o {
                                *live_uses.entry(*v).or_insert(0) += 1;
                            }
                        }
                    }
                }
                let mut removed_any = false;
                for blk in &mut m.functions[fi].blocks {
                    blk.instructions.retain(|inst| {
                        let dead = inst.result_id.is_some_and(|r| {
                            dce_candidates.contains(&r) && live_uses.get(&r).copied().unwrap_or(0) == 0
                        });
                        if dead {
                            removed_any = true;
                        }
                        !dead
                    });
                }
                if !removed_any {
                    break;
                }
            }
        }
    }

    lowered_total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{align_guard, buffer_addresses, table_ubo};
    use rspirv::dr::Module;
    use spirv_tools::{assembler::Assembler, val::Validator};

    fn assemble(text: &str) -> Vec<u8> {
        spirv_tools::assembler::create(Some(spirv_tools::TargetEnv::Vulkan_1_2))
            .assemble(text, Default::default())
            .unwrap()
            .as_bytes()
            .to_vec()
    }

    fn validate(bytes: &[u8]) {
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        let mut opts = spirv_tools::val::ValidatorOptions::default();
        opts.scalar_block_layout = true;
        spirv_tools::val::create(Some(spirv_tools::TargetEnv::Vulkan_1_2))
            .validate(&words, Some(opts))
            .unwrap();
    }

    const DYNAMIC_U16: &str = r#"
OpCapability Shader
OpCapability Int16
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %src %dst %gid
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %arr ArrayStride 4
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpDecorate %src DescriptorSet 0
OpDecorate %src Binding 0
OpDecorate %dst DescriptorSet 0
OpDecorate %dst Binding 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%ushort = OpTypeInt 16 0
%v2ushort = OpTypeVector %ushort 2
%v3uint = OpTypeVector %uint 3
%pin = OpTypePointer Input %v3uint
%gid = OpVariable %pin Input
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%c3 = OpConstant %uint 3
%c4 = OpConstant %uint 4
%arr = OpTypeRuntimeArray %uint
%block = OpTypeStruct %arr
%pb = OpTypePointer StorageBuffer %block
%pu = OpTypePointer StorageBuffer %uint
%src = OpVariable %pb StorageBuffer
%dst = OpVariable %pb StorageBuffer
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%g3 = OpLoad %v3uint %gid
%g = OpCompositeExtract %uint %g3 0
%byte = OpIMul %uint %g %c2
%w = OpUDiv %uint %byte %c4
%p = OpInBoundsAccessChain %pu %src %c0 %w
%word = OpLoad %uint %p
%m = OpBitwiseAnd %uint %byte %c3
%lane = OpShiftRightLogical %uint %m %c1
%vec = OpBitcast %v2ushort %word
%val = OpVectorExtractDynamic %ushort %vec %lane
%wide = OpUConvert %uint %val
%dp = OpInBoundsAccessChain %pu %dst %c0 %g
OpStore %dp %wide
OpReturn
OpFunctionEnd
"#;

    const DYNAMIC_U8: &str = r#"
OpCapability Shader
OpCapability Int8
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %src %dst %gid
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %arr ArrayStride 4
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpDecorate %src DescriptorSet 0
OpDecorate %src Binding 0
OpDecorate %dst DescriptorSet 0
OpDecorate %dst Binding 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uchar = OpTypeInt 8 0
%v4uchar = OpTypeVector %uchar 4
%v3uint = OpTypeVector %uint 3
%pin = OpTypePointer Input %v3uint
%gid = OpVariable %pin Input
%c0 = OpConstant %uint 0
%c3 = OpConstant %uint 3
%c4 = OpConstant %uint 4
%arr = OpTypeRuntimeArray %uint
%block = OpTypeStruct %arr
%pb = OpTypePointer StorageBuffer %block
%pu = OpTypePointer StorageBuffer %uint
%src = OpVariable %pb StorageBuffer
%dst = OpVariable %pb StorageBuffer
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%g3 = OpLoad %v3uint %gid
%g = OpCompositeExtract %uint %g3 0
%w = OpUDiv %uint %g %c4
%p = OpInBoundsAccessChain %pu %src %c0 %w
%word = OpLoad %uint %p
%lane = OpBitwiseAnd %uint %g %c3
%vec = OpBitcast %v4uchar %word
%val = OpVectorExtractDynamic %uchar %vec %lane
%wide = OpUConvert %uint %val
%dp = OpInBoundsAccessChain %pu %dst %c0 %g
OpStore %dp %wide
OpReturn
OpFunctionEnd
"#;

    const STATIC_HALF_LANE1: &str = r#"
OpCapability Shader
OpCapability Float16
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %src %dst %gid
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %arr ArrayStride 4
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpDecorate %src DescriptorSet 0
OpDecorate %src Binding 0
OpDecorate %dst DescriptorSet 0
OpDecorate %dst Binding 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%half = OpTypeFloat 16
%v2half = OpTypeVector %half 2
%v3uint = OpTypeVector %uint 3
%pin = OpTypePointer Input %v3uint
%gid = OpVariable %pin Input
%c0 = OpConstant %uint 0
%arr = OpTypeRuntimeArray %uint
%block = OpTypeStruct %arr
%pb = OpTypePointer StorageBuffer %block
%pu = OpTypePointer StorageBuffer %uint
%src = OpVariable %pb StorageBuffer
%dst = OpVariable %pb StorageBuffer
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%g3 = OpLoad %v3uint %gid
%g = OpCompositeExtract %uint %g3 0
%p = OpInBoundsAccessChain %pu %src %c0 %g
%word = OpLoad %uint %p
%vec = OpBitcast %v2half %word
%val = OpCompositeExtract %half %vec 1
%f = OpFConvert %float %val
%bits = OpBitcast %uint %f
%dp = OpInBoundsAccessChain %pu %dst %c0 %g
OpStore %dp %bits
OpReturn
OpFunctionEnd
"#;

    #[test]
    fn dynamic_u16_psb_load_lowers_to_native_16bit_load_without_udiv() {
        let input = assemble(DYNAMIC_U16);
        validate(&input);
        let lowered = buffer_addresses::lower(&input).unwrap();
        validate(&lowered);
        let ubo = table_ubo::convert(&lowered).unwrap();
        validate(&ubo);
        let (_, rep) = align_guard::guard(&ubo, false).unwrap();
        assert_eq!((rep.mask, rep.accesses), (0, 0), "natural align-2 load must not trigger align_guard: {rep:?}");

        let m: Module = rspirv::dr::load_bytes(&lowered).unwrap();
        let types: HashMap<u32, Instruction> = m
            .types_global_values
            .iter()
            .filter_map(|i| Some((i.result_id?, i.clone())))
            .collect();
        let body: Vec<&Instruction> = m.functions[0]
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .collect();
        assert!(
            body.iter().any(|i| {
                i.class.opcode == Op::Load
                    && i.result_type
                        .and_then(|t| types.get(&t))
                        .is_some_and(|t| {
                            t.class.opcode == Op::TypeInt
                                && t.operands == [Operand::LiteralBit32(16), Operand::LiteralBit32(0)]
                        })
                    && i.operands.get(2) == Some(&Operand::LiteralBit32(2))
            }),
            "expected native 16-bit Aligned 2 OpLoad"
        );
        assert!(
            !body.iter().any(|i| matches!(i.class.opcode, Op::UDiv | Op::VectorExtractDynamic)),
            "UDiv and VectorExtractDynamic must be eliminated"
        );
    }

    #[test]
    fn dynamic_u8_psb_load_lowers_to_native_8bit_load_without_udiv() {
        let input = assemble(DYNAMIC_U8);
        validate(&input);
        let lowered = buffer_addresses::lower(&input).unwrap();
        validate(&lowered);

        let m: Module = rspirv::dr::load_bytes(&lowered).unwrap();
        let types: HashMap<u32, Instruction> = m
            .types_global_values
            .iter()
            .filter_map(|i| Some((i.result_id?, i.clone())))
            .collect();
        let body: Vec<&Instruction> = m.functions[0]
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .collect();
        assert!(
            body.iter().any(|i| {
                i.class.opcode == Op::Load
                    && i.result_type
                        .and_then(|t| types.get(&t))
                        .is_some_and(|t| {
                            t.class.opcode == Op::TypeInt
                                && t.operands == [Operand::LiteralBit32(8), Operand::LiteralBit32(0)]
                        })
                    && i.operands.get(2) == Some(&Operand::LiteralBit32(1))
            }),
            "expected native 8-bit Aligned 1 OpLoad"
        );
        assert!(
            !body.iter().any(|i| matches!(i.class.opcode, Op::UDiv | Op::VectorExtractDynamic)),
            "UDiv and VectorExtractDynamic must be eliminated"
        );
    }

    #[test]
    fn static_half_lane1_lowers_to_native_half_load_with_byte_offset_2() {
        let input = assemble(STATIC_HALF_LANE1);
        validate(&input);
        let lowered = buffer_addresses::lower(&input).unwrap();
        validate(&lowered);

        let m: Module = rspirv::dr::load_bytes(&lowered).unwrap();
        let types: HashMap<u32, Instruction> = m
            .types_global_values
            .iter()
            .filter_map(|i| Some((i.result_id?, i.clone())))
            .collect();
        let body: Vec<&Instruction> = m.functions[0]
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .collect();
        assert!(
            body.iter().any(|i| {
                i.class.opcode == Op::Load
                    && i.result_type
                        .and_then(|t| types.get(&t))
                        .is_some_and(|t| {
                            t.class.opcode == Op::TypeFloat
                                && t.operands == [Operand::LiteralBit32(16)]
                        })
                    && i.operands.get(2) == Some(&Operand::LiteralBit32(2))
            }),
            "expected native half Aligned 2 OpLoad"
        );
        assert!(
            !body.iter().any(|i| i.class.opcode == Op::Bitcast
                && i.result_type
                    .and_then(|t| types.get(&t))
                    .is_some_and(|t| t.class.opcode == Op::TypeVector)),
            "vector bitcast for single-lane half load must be eliminated"
        );
    }

    #[test]
    fn multi_lane_half_extract_keeps_single_32bit_word_load() {
        let multi = STATIC_HALF_LANE1.replace(
            "%val = OpCompositeExtract %half %vec 1\n%f = OpFConvert %float %val",
            "%v0 = OpCompositeExtract %half %vec 0\n%v1 = OpCompositeExtract %half %vec 1\n%sum = OpFAdd %half %v0 %v1\n%f = OpFConvert %float %sum",
        );
        let input = assemble(&multi);
        validate(&input);
        let lowered = buffer_addresses::lower(&input).unwrap();
        validate(&lowered);

        let m: Module = rspirv::dr::load_bytes(&lowered).unwrap();
        let types: HashMap<u32, Instruction> = m
            .types_global_values
            .iter()
            .filter_map(|i| Some((i.result_id?, i.clone())))
            .collect();
        let body: Vec<&Instruction> = m.functions[0]
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .collect();
        assert!(
            !body.iter().any(|i| {
                i.class.opcode == Op::Load
                    && i.result_type
                        .and_then(|t| types.get(&t))
                        .is_some_and(|t| t.class.opcode == Op::TypeFloat)
            }),
            "multi-lane extract from same word must keep the single 32-bit word load"
        );
    }

    #[test]
    fn disabled_device_capability_keeps_word_extraction_path() {
        let input = assemble(DYNAMIC_U16);
        let lowered = buffer_addresses::lower_with_caps(
            &input,
            true,
            Caps { int8: true, int16: false, float16: true },
        )
        .unwrap();
        validate(&lowered);

        let m: Module = rspirv::dr::load_bytes(&lowered).unwrap();
        let body: Vec<&Instruction> = m.functions[0]
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .collect();
        assert!(
            body.iter().any(|i| i.class.opcode == Op::VectorExtractDynamic),
            "when device lacks shaderInt16, 16-bit subword load must stay on 32-bit word path"
        );
    }
}
