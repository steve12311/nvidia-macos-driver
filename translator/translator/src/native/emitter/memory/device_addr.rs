use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn emit_raw_load(
        &mut self,
        result: Word,
        ty: &LlType,
        raw: &RawBufferOffset,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        if raw.unmodelable {
            return Err("native emitter: raw buffer offset is not modelable".into());
        }
        if raw.device_addr_base.is_some() {
            let ty = self.resolve_type(ty)?;
            return self.emit_device_addr_load(result, &ty, raw, access_align, instructions);
        }
        self.emit_raw_load_at(result, ty, raw, 0, access_align, instructions)
    }

    fn wide_word_load_hint(
        &self,
        raw: &RawBufferOffset,
        extra: u64,
        access_align: Option<u64>,
    ) -> Option<u32> {
        if crate::env_vars::no_wide_half_load()
            || !matches!(
                self.raw_access_storage(raw).ok()?,
                StorageClass::UniformConstant | StorageClass::StorageBuffer
            )
        {
            return None;
        }
        let mut align = access_align?;
        if extra != 0 {
            align = align.min(1 << extra.trailing_zeros());
        }
        (align >= 8 && align.is_power_of_two()).then(|| align.min(16) as u32)
    }

    fn emit_raw_load_at(
        &mut self,
        result: Word,
        ty: &LlType,
        raw: &RawBufferOffset,
        extra: u64,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        match self.resolve_type(ty)? {
            LlType::Vector(elem, lanes) => {
                let elem = self.resolve_type(&elem)?;
                if matches!(elem, LlType::Half)
                    && lanes % 2 == 0
                    && self.raw_byte_array_root_len(raw).is_none()
                    && self.raw_word_access_can_use_word_index(raw, extra, access_align)
                    && std::env::var_os("NVMTL_NO_HALF_WORDS").is_none()
                {
                    let first = self.emit_raw_word_index_for_access(
                        raw,
                        extra,
                        access_align,
                        instructions,
                    )?;
                    let pair_ty = self.type_id(&LlType::Vector(Box::new(LlType::Half), 2))?;
                    let half_ty = self.type_id(&LlType::Half)?;
                    let mut lane_ids = Vec::with_capacity(lanes as usize);
                    for pair in 0..lanes / 2 {
                        let index = if pair == 0 {
                            first
                        } else {
                            self.emit_raw_word_index_for_access(
                                raw,
                                extra + 4 * u64::from(pair),
                                access_align,
                                instructions,
                            )?
                        };
                        let word = self.emit_raw_word_load_at_index(raw, index, instructions)?;
                        if pair + 1 < lanes / 2 {
                            if let Some(align) = self.wide_word_load_hint(
                                raw,
                                extra + 4 * u64::from(pair),
                                access_align,
                            ) {
                                if let Some(load) = instructions.last_mut().filter(|i| {
                                    i.class.opcode == Op::Load
                                        && i.result_id == Some(word)
                                        && i.operands.len() == 1
                                }) {
                                    load.operands.extend([
                                        Operand::MemoryAccess(spirv::MemoryAccess::ALIGNED),
                                        Operand::LiteralBit32(align),
                                    ]);
                                }
                            }
                        }
                        let both = self.fresh();
                        instructions.push(Self::inst(
                            Op::Bitcast,
                            Some(pair_ty),
                            Some(both),
                            vec![Operand::IdRef(word)],
                        ));
                        for slot in 0..2u32 {
                            let id = self.fresh();
                            instructions.push(Self::inst(
                                Op::CompositeExtract,
                                Some(half_ty),
                                Some(id),
                                vec![Operand::IdRef(both), Operand::LiteralBit32(slot)],
                            ));
                            lane_ids.push(id);
                        }
                    }
                    let result_type = self.type_id(&LlType::Vector(Box::new(elem), lanes))?;
                    instructions.push(Self::inst(
                        Op::CompositeConstruct,
                        Some(result_type),
                        Some(result),
                        lane_ids.into_iter().map(Operand::IdRef).collect(),
                    ));
                    return Ok(());
                }
                let (elem_size, _) = self.raw_type_size_align(&elem)?;
                let wide_lanes = elem_size == 4
                    && matches!(elem, LlType::Float | LlType::Int(32))
                    && lanes >= 2
                    && self.raw_byte_array_root_len(raw).is_none();
                let uint_ty = if wide_lanes {
                    Some(self.type_id(&LlType::Int(32))?)
                } else {
                    None
                };
                let mut lane_ids = Vec::with_capacity(lanes as usize);
                for lane in 0..lanes {
                    let lane_id = self.fresh();
                    let start = instructions.len();
                    self.emit_raw_scalar_load(
                        lane_id,
                        &elem,
                        raw,
                        extra + lane as u64 * elem_size,
                        access_align,
                        instructions,
                    )?;
                    if let Some(uint_ty) = uint_ty.filter(|_| lane % 2 == 0 && lane + 1 < lanes) {
                        let at = extra + lane as u64 * elem_size;
                        if self.raw_word_access_can_use_word_index(raw, at, access_align) {
                            if let Some(align) = self.wide_word_load_hint(raw, at, access_align) {
                                let mut words = instructions[start..].iter_mut().filter(|i| {
                                    i.class.opcode == Op::Load
                                        && i.result_type == Some(uint_ty)
                                        && i.operands.len() == 1
                                });
                                if let (Some(load), None) = (words.next(), words.next()) {
                                    load.operands.extend([
                                        Operand::MemoryAccess(spirv::MemoryAccess::ALIGNED),
                                        Operand::LiteralBit32(align),
                                    ]);
                                }
                            }
                        }
                    }
                    lane_ids.push(lane_id);
                }
                let result_type = self.type_id(&LlType::Vector(Box::new(elem), lanes))?;
                instructions.push(Self::inst(
                    Op::CompositeConstruct,
                    Some(result_type),
                    Some(result),
                    lane_ids.into_iter().map(Operand::IdRef).collect(),
                ));
            }
            LlType::Array(elem, len) => {
                let elem = self.resolve_type(&elem)?;
                let (elem_size, elem_align) = self.raw_type_size_align(&elem)?;
                let stride = elem_size.div_ceil(elem_align) * elem_align;
                let mut ids = Vec::with_capacity(len as usize);
                for i in 0..len {
                    let id = self.fresh();
                    self.emit_raw_load_at(
                        id,
                        &elem,
                        raw,
                        extra + i as u64 * stride,
                        access_align,
                        instructions,
                    )?;
                    ids.push(id);
                }
                let result_type = self.type_id(&LlType::Array(Box::new(elem), len))?;
                instructions.push(Self::inst(
                    Op::CompositeConstruct,
                    Some(result_type),
                    Some(result),
                    ids.into_iter().map(Operand::IdRef).collect(),
                ));
            }
            LlType::Struct(fields) => {
                let mut off = 0u64;
                let mut ids = Vec::with_capacity(fields.len());
                for field in &fields {
                    let (size, align) = self.raw_type_size_align(field)?;
                    off = off.div_ceil(align) * align;
                    let id = self.fresh();
                    self.emit_raw_load_at(id, field, raw, extra + off, access_align, instructions)?;
                    ids.push(id);
                    off += size;
                }
                let result_type = self.type_id(&LlType::Struct(fields))?;
                instructions.push(Self::inst(
                    Op::CompositeConstruct,
                    Some(result_type),
                    Some(result),
                    ids.into_iter().map(Operand::IdRef).collect(),
                ));
            }
            scalar => {
                self.emit_raw_scalar_load(result, &scalar, raw, extra, access_align, instructions)?
            }
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn can_materialize_bda_pointer_address(
        &self,
        value: &LlValue,
    ) -> bool {
        if !self.bda_device_pointers {
            return false;
        }
        match value {
            LlValue::Zero | LlValue::Undef => true,
            LlValue::Local(name) => {
                self.bda_direct_addresses.contains_key(name)
                    || self.direct_param_indices.contains_key(name)
                    || (self.values.contains_key(name)
                        && self.pointer_storage.get(name)
                            == Some(&StorageClass::PhysicalStorageBuffer))
                    || self.raw_offsets.get(name).is_some_and(|raw| {
                        !raw.unmodelable
                            && (raw.device_addr_base.is_some()
                                || self.bda_direct_addresses.contains_key(&raw.root)
                                || self.direct_param_indices.contains_key(&raw.root))
                    })
            }
            _ => false,
        }
    }

    pub(in crate::native::emitter) fn materialize_bda_pointer_address(
        &mut self,
        value: &LlValue,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<Word>, String> {
        if !self.bda_device_pointers {
            return Ok(None);
        }
        match value {
            LlValue::Zero | LlValue::Undef => Ok(Some(self.const_signed_int(64, 0)?)),
            LlValue::Local(name) => {
                if let Some(raw) = self.raw_offsets.get(name).cloned() {
                    if !raw.unmodelable
                        && (raw.device_addr_base.is_some()
                            || self.bda_direct_addresses.contains_key(&raw.root)
                            || self.direct_param_indices.contains_key(&raw.root))
                    {
                        if raw.const_off == 0 && raw.dyn_terms.is_empty() {
                            if let Some(addr) = self.bda_direct_addresses.get(name).copied() {
                                return Ok(Some(addr));
                            }
                        }
                        self.used_device_address = true;
                        return Ok(Some(self.materialize_device_address(&raw, instructions)?));
                    }
                }
                if let Some(addr) = self.bda_direct_addresses.get(name).copied() {
                    self.used_device_address = true;
                    return Ok(Some(addr));
                }
                if self.direct_param_indices.contains_key(name) {
                    let (low, high) = self.emit_direct_buffer_address_payload(name, instructions)?;
                    self.used_device_address = true;
                    return Ok(Some(
                        self.combine_pointer_payload_words(low, high, instructions)?,
                    ));
                }
                if self.pointer_storage.get(name) == Some(&StorageClass::PhysicalStorageBuffer) {
                    if let Some(ptr_id) = self.values.get(name).map(|(id, _)| *id) {
                        let i64_ty = self.type_id(&LlType::Int(64))?;
                        let addr = self.fresh();
                        instructions.push(Self::inst(
                            Op::ConvertPtrToU,
                            Some(i64_ty),
                            Some(addr),
                            vec![Operand::IdRef(ptr_id)],
                        ));
                        self.used_device_address = true;
                        return Ok(Some(addr));
                    }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    pub(in crate::native::emitter) fn materialize_device_address(
        &mut self,
        raw: &RawBufferOffset,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let base = if let Some(base) = raw.device_addr_base {
            base
        } else if let Some(base) = self.bda_direct_addresses.get(&raw.root).copied() {
            base
        } else if self.direct_param_indices.contains_key(&raw.root) {
            let (low, high) = self.emit_direct_buffer_address_payload(&raw.root, instructions)?;
            self.combine_pointer_payload_words(low, high, instructions)?
        } else {
            return Err("native emitter: device address offset has no base".to_string());
        };
        let i64_ty = self.type_id(&LlType::Int(64))?;
        let mut addr = base;
        if raw.const_off != 0 {
            let c = self.const_signed_int(64, raw.const_off)?;
            let sum = self.fresh();
            instructions.push(Self::inst(
                Op::IAdd,
                Some(i64_ty),
                Some(sum),
                vec![Operand::IdRef(addr), Operand::IdRef(c)],
            ));
            addr = sum;
        }
        let terms = raw.dyn_terms.clone();
        for (tv, stride) in &terms {
            let idx = self.value_id_in(&tv.value, &tv.ty, instructions)?;
            let idx64 = match self.resolve_type(&tv.ty)? {
                LlType::Int(64) => idx,
                _ => {
                    let w = self.fresh();
                    instructions.push(Self::inst(
                        Op::SConvert,
                        Some(i64_ty),
                        Some(w),
                        vec![Operand::IdRef(idx)],
                    ));
                    w
                }
            };
            let term = if *stride == 1 {
                idx64
            } else {
                let s = self.const_signed_int(64, *stride)?;
                let m = self.fresh();
                instructions.push(Self::inst(
                    Op::IMul,
                    Some(i64_ty),
                    Some(m),
                    vec![Operand::IdRef(idx64), Operand::IdRef(s)],
                ));
                m
            };
            let sum = self.fresh();
            instructions.push(Self::inst(
                Op::IAdd,
                Some(i64_ty),
                Some(sum),
                vec![Operand::IdRef(addr), Operand::IdRef(term)],
            ));
            addr = sum;
        }
        Ok(addr)
    }

    pub(in crate::native::emitter) fn materialize_reserved_bda_address(
        &mut self,
        name: &str,
        raw: &RawBufferOffset,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let address_name = bda_address_name(name);
        let Some((reserved, _)) = self.values.get(&address_name).cloned() else {
            return Ok(());
        };
        let address = self.materialize_device_address(raw, instructions)?;
        if address != reserved {
            let address_ty = self.type_id(&LlType::Int(64))?;
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(address_ty),
                Some(reserved),
                vec![Operand::IdRef(address)],
            ));
        }
        self.bda_address_values.insert(reserved);
        Ok(())
    }

    pub(in crate::native::emitter) fn device_addr_align(&mut self, ty: &LlType) -> u32 {
        let scalar = match ty {
            LlType::Vector(elem, _) => elem.as_ref().clone(),
            other => other.clone(),
        };
        match scalar {
            LlType::Int(64) | LlType::Ptr(_) => 8,
            LlType::Int(32) | LlType::Float => 4,
            LlType::Int(16) | LlType::Half | LlType::BFloat => 2,
            LlType::Int(8) => 1,
            _ => 4,
        }
    }

    pub(in crate::native::emitter) fn device_addr_align_for_access(
        &mut self,
        ty: &LlType,
        access_align: Option<u64>,
    ) -> u32 {
        let natural = self.device_addr_align(ty);
        match access_align {
            Some(a) if a > 0 && a.is_power_of_two() => {
                natural.min(a.min(u32::MAX as u64) as u32)
            }
            _ => natural,
        }
    }

    pub(in crate::native::emitter) fn emit_device_addr_load(
        &mut self,
        result: Word,
        ty: &LlType,
        raw: &RawBufferOffset,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        self.used_device_address = true;
        let addr = self.materialize_device_address(raw, instructions)?;
        let base = raw
            .device_addr_base
            .ok_or("native emitter: device load has no base")?;
        let result_ty = self.type_id(ty)?;
        let align = self.device_addr_align_for_access(ty, access_align);
        let helper = self.null_bda_load_helper(ty, align)?;
        instructions.push(Self::inst(
            Op::FunctionCall,
            Some(result_ty),
            Some(result),
            vec![
                Operand::IdRef(helper),
                Operand::IdRef(base),
                Operand::IdRef(addr),
            ],
        ));
        Ok(())
    }

    fn null_bda_load_helper(&mut self, ty: &LlType, align: u32) -> Result<Word, String> {
        let key = (ty.clone(), self.constant_space_load, align);
        if let Some(helper) = self.null_bda_load_helpers.get(&key) {
            return Ok(*helper);
        }
        let result_ty = self.type_id(ty)?;
        let address_ty = self.type_id(&LlType::Int(64))?;
        let bool_ty = self.type_id(&LlType::Bool)?;
        let zero_address = self.const_signed_int(64, 0)?;
        let zero_result = self.const_null(ty)?;
        let function_type = self.function_type_id(result_ty, &[address_ty, address_ty]);
        let helper = self.fresh();
        let base = self.fresh();
        let addr = self.fresh();
        let entry = self.fresh();
        let live = self.fresh();
        let merge = self.fresh();
        let valid = self.fresh();
        let result = self.fresh();
        let entry_instructions = vec![
            Self::inst(
                Op::INotEqual,
                Some(bool_ty),
                Some(valid),
                vec![Operand::IdRef(base), Operand::IdRef(zero_address)],
            ),
            Self::inst(
                Op::SelectionMerge,
                None,
                None,
                vec![
                    Operand::IdRef(merge),
                    Operand::SelectionControl(SelectionControl::NONE),
                ],
            ),
            Self::inst(
                Op::BranchConditional,
                None,
                None,
                vec![
                    Operand::IdRef(valid),
                    Operand::IdRef(live),
                    Operand::IdRef(merge),
                ],
            ),
        ];
        let mut instructions = Vec::new();
        let loaded = self.emit_psb_load_value(ty, addr, align, &mut instructions)?;
        instructions.push(Self::inst(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(merge)],
        ));
        self.module.functions.push(Function {
            def: Some(Self::inst(
                Op::Function,
                Some(result_ty),
                Some(helper),
                vec![
                    Operand::FunctionControl(FunctionControl::INLINE),
                    Operand::IdRef(function_type),
                ],
            )),
            end: Some(Self::inst(Op::FunctionEnd, None, None, vec![])),
            parameters: vec![
                Self::inst(Op::FunctionParameter, Some(address_ty), Some(base), vec![]),
                Self::inst(Op::FunctionParameter, Some(address_ty), Some(addr), vec![]),
            ],
            blocks: vec![
                Block {
                    label: Some(Self::inst(Op::Label, None, Some(entry), vec![])),
                    instructions: entry_instructions,
                },
                Block {
                    label: Some(Self::inst(Op::Label, None, Some(live), vec![])),
                    instructions,
                },
                Block {
                    label: Some(Self::inst(Op::Label, None, Some(merge), vec![])),
                    instructions: vec![
                        Self::inst(
                            Op::Phi,
                            Some(result_ty),
                            Some(result),
                            vec![
                                Operand::IdRef(zero_result),
                                Operand::IdRef(entry),
                                Operand::IdRef(loaded),
                                Operand::IdRef(live),
                            ],
                        ),
                        Self::inst(Op::ReturnValue, None, None, vec![Operand::IdRef(result)]),
                    ],
                },
            ],
        });
        self.null_bda_load_helpers.insert(key, helper);
        Ok(helper)
    }

    fn nonwritable_psb_wrapper(&mut self, ty: &LlType) -> Result<Option<Word>, String> {
        if !self.constant_space_load || std::env::var_os("NVMTL_NO_CONST_NONWRITABLE").is_some() {
            return Ok(None);
        }
        let plain = |t: &LlType| {
            matches!(
                t,
                LlType::Int(8 | 16 | 32 | 64) | LlType::Float | LlType::Half
            )
        };
        let wrappable = match ty {
            LlType::Vector(elem, lanes) => (2..=4).contains(lanes) && plain(elem.as_ref()),
            other => plain(other),
        };
        if !wrappable {
            return Ok(None);
        }
        if let Some(id) = self.nonwritable_psb_wrappers.get(ty) {
            return Ok(Some(*id));
        }
        let member_ty = self.resolve_type(ty)?;
        let member = self.type_id(&member_ty)?;
        let wrapper = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::TypeStruct,
            None,
            Some(wrapper),
            vec![Operand::IdRef(member)],
        ));
        self.module.annotations.push(Self::inst(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(wrapper),
                Operand::Decoration(spirv::Decoration::Block),
            ],
        ));
        self.module.annotations.push(Self::inst(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(wrapper),
                Operand::LiteralBit32(0),
                Operand::Decoration(spirv::Decoration::Offset),
                Operand::LiteralBit32(0),
            ],
        ));
        self.module.annotations.push(Self::inst(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(wrapper),
                Operand::LiteralBit32(0),
                Operand::Decoration(spirv::Decoration::NonWritable),
            ],
        ));
        let wrapper_ptr = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::TypePointer,
            None,
            Some(wrapper_ptr),
            vec![
                Operand::StorageClass(StorageClass::PhysicalStorageBuffer),
                Operand::IdRef(wrapper),
            ],
        ));
        self.nonwritable_psb_wrappers
            .insert(ty.clone(), wrapper_ptr);
        Ok(Some(wrapper_ptr))
    }

    fn emit_psb_addr_offset(
        &mut self,
        addr: Word,
        byte_offset: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        if byte_offset == 0 {
            return Ok(addr);
        }
        let i64_ty = self.type_id(&LlType::Int(64))?;
        let off = self.const_signed_int(64, byte_offset as i64)?;
        let out = self.fresh();
        instructions.push(Self::inst(
            Op::IAdd,
            Some(i64_ty),
            Some(out),
            vec![Operand::IdRef(addr), Operand::IdRef(off)],
        ));
        Ok(out)
    }

    fn emit_psb_aligned_load(
        &mut self,
        ty: &LlType,
        addr: Word,
        align: u32,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let result_ty = self.type_id(ty)?;
        let ptr_ty = self.ptr_type_id(StorageClass::PhysicalStorageBuffer, ty)?;
        let p = match self.nonwritable_psb_wrapper(ty)? {
            Some(wrapper_ptr_ty) => {
                let w = self.fresh();
                instructions.push(Self::inst(
                    Op::ConvertUToPtr,
                    Some(wrapper_ptr_ty),
                    Some(w),
                    vec![Operand::IdRef(addr)],
                ));
                let zero = self.const_uint(0)?;
                let p = self.fresh();
                instructions.push(Self::inst(
                    Op::AccessChain,
                    Some(ptr_ty),
                    Some(p),
                    vec![Operand::IdRef(w), Operand::IdRef(zero)],
                ));
                p
            }
            None => {
                let p = self.fresh();
                instructions.push(Self::inst(
                    Op::ConvertUToPtr,
                    Some(ptr_ty),
                    Some(p),
                    vec![Operand::IdRef(addr)],
                ));
                p
            }
        };
        let loaded = self.fresh();
        instructions.push(Self::inst(
            Op::Load,
            Some(result_ty),
            Some(loaded),
            vec![
                Operand::IdRef(p),
                Operand::MemoryAccess(spirv::MemoryAccess::ALIGNED),
                Operand::LiteralBit32(align),
            ],
        ));
        Ok(loaded)
    }

    // NVK/NAK requires every PhysicalStorageBuffer load/store to satisfy size_B <= align.
    // When LLVM specifies a smaller alignment (e.g. 2-byte aligned i32 in 22-byte block_q5_0),
    // split the access into naturally aligned chunks of size `align`.
    fn emit_psb_load_value(
        &mut self,
        ty: &LlType,
        addr: Word,
        align: u32,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let natural = self.device_addr_align(ty);
        if align >= natural {
            return self.emit_psb_aligned_load(ty, addr, align, instructions);
        }
        if let LlType::Vector(elem, lanes) = ty {
            let elem_bytes = u64::from(natural);
            let mut components = Vec::with_capacity(*lanes as usize);
            for lane in 0..*lanes {
                let lane_addr =
                    self.emit_psb_addr_offset(addr, u64::from(lane) * elem_bytes, instructions)?;
                components.push(self.emit_psb_load_value(elem, lane_addr, align, instructions)?);
            }
            let vec_ty = self.type_id(ty)?;
            let composite = self.fresh();
            instructions.push(Self::inst(
                Op::CompositeConstruct,
                Some(vec_ty),
                Some(composite),
                components.into_iter().map(Operand::IdRef).collect(),
            ));
            return Ok(composite);
        }
        if !matches!(align, 1 | 2 | 4) || !matches!(natural, 2 | 4 | 8) || natural % align != 0 {
            return self.emit_psb_aligned_load(ty, addr, align, instructions);
        }
        let chunk_ll = LlType::Int(align * 8);
        let int_ll = LlType::Int(natural * 8);
        let int_ty = self.type_id(&int_ll)?;
        let count = natural / align;
        let mut acc = 0 as Word;
        for k in 0..count {
            let chunk_addr =
                self.emit_psb_addr_offset(addr, u64::from(k * align), instructions)?;
            let chunk = self.emit_psb_aligned_load(&chunk_ll, chunk_addr, align, instructions)?;
            let widened = self.fresh();
            instructions.push(Self::inst(
                Op::UConvert,
                Some(int_ty),
                Some(widened),
                vec![Operand::IdRef(chunk)],
            ));
            let piece = if k == 0 {
                widened
            } else {
                let shift = self.const_uint(k * align * 8)?;
                let shifted = self.fresh();
                instructions.push(Self::inst(
                    Op::ShiftLeftLogical,
                    Some(int_ty),
                    Some(shifted),
                    vec![Operand::IdRef(widened), Operand::IdRef(shift)],
                ));
                shifted
            };
            acc = if k == 0 {
                piece
            } else {
                let combined = self.fresh();
                instructions.push(Self::inst(
                    Op::BitwiseOr,
                    Some(int_ty),
                    Some(combined),
                    vec![Operand::IdRef(acc), Operand::IdRef(piece)],
                ));
                combined
            };
        }
        let result_ty = self.type_id(ty)?;
        if result_ty == int_ty {
            return Ok(acc);
        }
        let op = if matches!(ty, LlType::Ptr(_)) {
            Op::ConvertUToPtr
        } else {
            Op::Bitcast
        };
        let cast = self.fresh();
        instructions.push(Self::inst(
            op,
            Some(result_ty),
            Some(cast),
            vec![Operand::IdRef(acc)],
        ));
        Ok(cast)
    }

    fn emit_psb_aligned_store(
        &mut self,
        ty: &LlType,
        value: Word,
        addr: Word,
        align: u32,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let ptr_ty = self.ptr_type_id(StorageClass::PhysicalStorageBuffer, ty)?;
        let p = self.fresh();
        instructions.push(Self::inst(
            Op::ConvertUToPtr,
            Some(ptr_ty),
            Some(p),
            vec![Operand::IdRef(addr)],
        ));
        instructions.push(Self::inst(
            Op::Store,
            None,
            None,
            vec![
                Operand::IdRef(p),
                Operand::IdRef(value),
                Operand::MemoryAccess(spirv::MemoryAccess::ALIGNED),
                Operand::LiteralBit32(align),
            ],
        ));
        Ok(())
    }

    fn emit_psb_store_value(
        &mut self,
        ty: &LlType,
        value: Word,
        addr: Word,
        align: u32,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let natural = self.device_addr_align(ty);
        if align >= natural {
            return self.emit_psb_aligned_store(ty, value, addr, align, instructions);
        }
        if let LlType::Vector(elem, lanes) = ty {
            let elem_ty = self.type_id(elem)?;
            let elem_bytes = u64::from(natural);
            for lane in 0..*lanes {
                let comp = self.fresh();
                instructions.push(Self::inst(
                    Op::CompositeExtract,
                    Some(elem_ty),
                    Some(comp),
                    vec![Operand::IdRef(value), Operand::LiteralBit32(lane)],
                ));
                let lane_addr =
                    self.emit_psb_addr_offset(addr, u64::from(lane) * elem_bytes, instructions)?;
                self.emit_psb_store_value(elem, comp, lane_addr, align, instructions)?;
            }
            return Ok(());
        }
        if !matches!(align, 1 | 2 | 4) || !matches!(natural, 2 | 4 | 8) || natural % align != 0 {
            return self.emit_psb_aligned_store(ty, value, addr, align, instructions);
        }
        let chunk_ll = LlType::Int(align * 8);
        let chunk_ty = self.type_id(&chunk_ll)?;
        let int_ll = LlType::Int(natural * 8);
        let int_ty = self.type_id(&int_ll)?;
        let value_ty = self.type_id(ty)?;
        let int_val = if value_ty == int_ty {
            value
        } else {
            let op = if matches!(ty, LlType::Ptr(_)) {
                Op::ConvertPtrToU
            } else {
                Op::Bitcast
            };
            let cast = self.fresh();
            instructions.push(Self::inst(
                op,
                Some(int_ty),
                Some(cast),
                vec![Operand::IdRef(value)],
            ));
            cast
        };
        let count = natural / align;
        for k in 0..count {
            let shifted = if k == 0 {
                int_val
            } else {
                let shift = self.const_uint(k * align * 8)?;
                let s = self.fresh();
                instructions.push(Self::inst(
                    Op::ShiftRightLogical,
                    Some(int_ty),
                    Some(s),
                    vec![Operand::IdRef(int_val), Operand::IdRef(shift)],
                ));
                s
            };
            let chunk = self.fresh();
            instructions.push(Self::inst(
                Op::UConvert,
                Some(chunk_ty),
                Some(chunk),
                vec![Operand::IdRef(shifted)],
            ));
            let chunk_addr =
                self.emit_psb_addr_offset(addr, u64::from(k * align), instructions)?;
            self.emit_psb_aligned_store(&chunk_ll, chunk, chunk_addr, align, instructions)?;
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn emit_device_addr_store(
        &mut self,
        ty: &LlType,
        value: Word,
        raw: &RawBufferOffset,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        self.used_device_address = true;
        let addr = self.materialize_device_address(raw, instructions)?;
        let align = self.device_addr_align_for_access(ty, access_align);
        self.emit_psb_store_value(ty, value, addr, align, instructions)
    }
}
