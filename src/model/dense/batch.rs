use super::{Layer, Timings, elapsed};
use crate::model::{checkpoint::Tensor, cpu::Cpu, trace::Trace};
use anyhow::Result;
use std::time::Instant;

#[derive(Clone, Copy)]
pub(super) struct Geometry {
    pub h: usize,
    pub f: usize,
    pub qh: usize,
    pub kh: usize,
    pub d: usize,
    pub epsilon: f32,
    pub embedding_multiplier: f32,
    pub residual_multiplier: f32,
}
pub(super) struct Batch {
    pub rows: usize,
    pub position: usize,
    pub hidden: Vec<f32>,
    pub normalized: Vec<f32>,
    pub projected: Vec<f32>,
    pub qkv: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub attended: Vec<f32>,
    pub gate_up: Vec<f32>,
    pub middle: Vec<f32>,
    pub scatter: Vec<f32>,
}
impl Batch {
    pub fn new(rows: usize, g: Geometry, scatter: usize) -> Self {
        Self {
            rows: 0,
            position: 0,
            hidden: vec![0.; rows * g.h],
            normalized: vec![0.; rows * g.h],
            projected: vec![0.; rows * g.h],
            qkv: vec![0.; rows * (g.qh + 2 * g.kh) * g.d],
            query: vec![0.; rows * g.qh * g.d],
            key: vec![0.; rows * g.kh * g.d],
            value: vec![0.; rows * g.kh * g.d],
            attended: vec![0.; rows * g.qh * g.d],
            gate_up: vec![0.; rows * 2 * g.f],
            middle: vec![0.; rows * g.f],
            scatter: vec![0.; rows * scatter],
        }
    }
    pub fn bytes(&self) -> usize {
        [
            &self.hidden,
            &self.normalized,
            &self.projected,
            &self.qkv,
            &self.query,
            &self.key,
            &self.value,
            &self.attended,
            &self.gate_up,
            &self.middle,
            &self.scatter,
        ]
        .iter()
        .map(|v| v.capacity() * 4)
        .sum()
    }
    pub fn embed(
        &mut self,
        tokens: &[u32],
        position: usize,
        tensor: &Tensor,
        cpu: &Cpu,
        g: Geometry,
    ) {
        self.rows = tokens.len();
        self.position = position;
        for (r, token) in tokens.iter().enumerate() {
            tensor.row(*token as usize, &mut self.hidden[r * g.h..(r + 1) * g.h]);
        }
        cpu.scale(&mut self.hidden[..self.rows * g.h], g.embedding_multiplier);
    }
    fn trace(&self, trace: &Trace, index: i32, stage: &str, values: &[f32]) -> Result<()> {
        trace.save(self.position, index, stage, values)
    }
}

pub(super) struct LayerPass<'a> {
    pub cpu: &'a Cpu,
    pub trace: &'a Trace,
    pub cosine: &'a [f32],
    pub sine: &'a [f32],
    pub geometry: Geometry,
}
impl LayerPass<'_> {
    pub fn run(
        &self,
        layer: &Layer,
        index: i32,
        b: &mut Batch,
        timings: &mut Timings,
    ) -> Result<()> {
        let g = self.geometry;
        let rows = b.rows;
        let (h, d) = (g.h, g.d);
        let (q_width, kv_width) = (g.qh * d, g.kh * d);
        let qkv_width = q_width + 2 * kv_width;
        let mut start;
        start = Instant::now();
        self.cpu.norm(
            &b.hidden[..rows * h],
            &mut b.normalized[..rows * h],
            &layer.input_norm,
            g.epsilon,
        );
        timings.cpu += elapsed(start);
        b.trace(self.trace, index, "input", &b.normalized[..rows * h])?;
        start = Instant::now();
        layer.qkv.run(
            &b.normalized[..rows * h],
            &mut b.qkv[..rows * qkv_width],
            rows,
            &mut b.scatter,
        )?;
        timings.linear += elapsed(start);
        b.trace(self.trace, index, "qkv", &b.qkv[..rows * qkv_width])?;
        start = Instant::now();
        for r in 0..rows {
            let source = &b.qkv[r * qkv_width..(r + 1) * qkv_width];
            b.query[r * q_width..(r + 1) * q_width].copy_from_slice(&source[..q_width]);
            b.key[r * kv_width..(r + 1) * kv_width]
                .copy_from_slice(&source[q_width..q_width + kv_width]);
            b.value[r * kv_width..(r + 1) * kv_width]
                .copy_from_slice(&source[q_width + kv_width..]);
        }
        self.cpu.rope(
            &mut b.query[..rows * q_width],
            g.qh,
            d,
            &layer.q_norm,
            g.epsilon,
            self.cosine,
            self.sine,
            b.position,
        );
        self.cpu.rope(
            &mut b.key[..rows * kv_width],
            g.kh,
            d,
            &layer.k_norm,
            g.epsilon,
            self.cosine,
            self.sine,
            b.position,
        );
        timings.cpu += elapsed(start);
        b.trace(self.trace, index, "query", &b.query[..rows * q_width])?;
        b.trace(self.trace, index, "key", &b.key[..rows * kv_width])?;
        b.trace(self.trace, index, "value", &b.value[..rows * kv_width])?;
        start = Instant::now();
        layer.attention.run(
            &b.query[..rows * q_width],
            &b.key[..rows * kv_width],
            &b.value[..rows * kv_width],
            &mut b.attended[..rows * q_width],
            rows,
            b.position,
        )?;
        timings.attention += elapsed(start);
        b.trace(self.trace, index, "attn", &b.attended[..rows * q_width])?;
        start = Instant::now();
        layer.o.run(
            &b.attended[..rows * q_width],
            &mut b.projected[..rows * h],
            rows,
        )?;
        timings.linear += elapsed(start);
        b.trace(self.trace, index, "o", &b.projected[..rows * h])?;
        start = Instant::now();
        self.cpu
            .scale(&mut b.projected[..rows * h], g.residual_multiplier);
        self.cpu
            .add(&mut b.hidden[..rows * h], &b.projected[..rows * h]);
        self.cpu.norm(
            &b.hidden[..rows * h],
            &mut b.normalized[..rows * h],
            &layer.post_norm,
            g.epsilon,
        );
        timings.cpu += elapsed(start);
        b.trace(self.trace, index, "post", &b.normalized[..rows * h])?;
        start = Instant::now();
        if !self.trace.enabled()
            && layer.gate_up.run_swiglu(
                &b.normalized[..rows * h],
                &mut b.middle[..rows * g.f],
                rows,
            )?
        {
            timings.linear += elapsed(start);
        } else {
            layer.gate_up.run(
                &b.normalized[..rows * h],
                &mut b.gate_up[..rows * 2 * g.f],
                rows,
                &mut b.scatter,
            )?;
            timings.linear += elapsed(start);
            b.trace(self.trace, index, "gate_up", &b.gate_up[..rows * 2 * g.f])?;
            start = Instant::now();
            self.cpu.swiglu(
                &b.gate_up[..rows * 2 * g.f],
                &mut b.middle[..rows * g.f],
                g.f,
            );
            timings.cpu += elapsed(start);
        }
        b.trace(self.trace, index, "middle", &b.middle[..rows * g.f])?;
        start = Instant::now();
        layer
            .down
            .run(&b.middle[..rows * g.f], &mut b.projected[..rows * h], rows)?;
        timings.linear += elapsed(start);
        b.trace(self.trace, index, "down", &b.projected[..rows * h])?;
        start = Instant::now();
        self.cpu
            .scale(&mut b.projected[..rows * h], g.residual_multiplier);
        self.cpu
            .add(&mut b.hidden[..rows * h], &b.projected[..rows * h]);
        timings.cpu += elapsed(start);
        b.trace(self.trace, index, "hidden", &b.hidden[..rows * h])?;
        Ok(())
    }
}
