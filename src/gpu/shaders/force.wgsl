// Sums the momentum-exchange forces of one body per workgroup (body slot = workgroup + 1) over
// the list of fluid nodes next to solids, in float32 with a shared-memory tree, and writes the
// totals for the current step into a ring buffer that the CPU reads back every few frames.

struct ReduceParams {
  listCount: u32,
  histLen: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> R: ReduceParams;
@group(0) @binding(1) var<storage, read> list: array<u32>;
@group(0) @binding(2) var<storage, read> nodeForce: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> history: array<f32>;

var<workgroup> sumX: array<f32, 256>;
var<workgroup> sumY: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let slot = wid.x + 1u;
  var ax = 0.0;
  var ay = 0.0;
  for (var k = li; k < R.listCount; k += 256u) {
    let idx = list[k];
    let s = atomicLoad(&status[4u + idx]);
    let f = nodeForce[idx];
    if ((s & 0xffu) == slot) {
      ax += f.x;
      ay += f.y;
    }
    if (((s >> 8u) & 0xffu) == slot) {
      ax += f.z;
      ay += f.w;
    }
  }
  sumX[li] = ax;
  sumY[li] = ay;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (li < stride) {
      sumX[li] += sumX[li + stride];
      sumY[li] += sumY[li + stride];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let step = atomicLoad(&status[1]);
    let row = ((step + R.histLen - 1u) % R.histLen) * 32u;
    history[row + 2u * wid.x] = sumX[0];
    history[row + 2u * wid.x + 1u] = sumY[0];
  }
}
