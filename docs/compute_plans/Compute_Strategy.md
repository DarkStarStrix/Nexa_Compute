# Atheron Labs — Compute Strategy
### Foundation Model Training & HPC Operations (v1.0)

This document defines the **compute strategy** for Atheron Labs' molecular foundation model program.  
It describes *how compute will be used*, *how risk is reduced*, and *how training stability and throughput are maximized*.  
It is not a schedule or a plan — it is a **doctrine**.

---

## 1. Principles

### **1.1 Compute Is Cheap; Mistakes Are Expensive**
H100 time is affordable relative to:
- engineering throughput
- model quality
- risk of multi-day failures
- downstream work blocked by errors

**Strategy:** prioritize correctness, reproducibility, and stability over small GPU savings.

---

### **1.2 Pre-Flight Before Burn**
Never launch a multi-week run without a complete cluster shakeout.

Three validation tiers:
1. **V1 – Stability:** ensure cluster, NCCL, FSDP, and dataloaders work.
2. **V2 – Throughput:** tune microbatch, grad-accum, and IO saturation.
3. **V3 – Dress Rehearsal:** run the full architecture in short form.

Only after V3 passes at ≥75% utilization does full training begin.

---

### **1.3 Never Debug at Scale**
All debugging must be done:
- with reduced model size  
- with reduced data  
- with reduced step counts  
- on a single node when possible

Multi-node debugging is only done *after single-node stability is proven*.

---

## 2. Training Compute Strategy

### **2.1 Use NVMe as the Hot Path**
All shards must be:
- copied from Wasabi → NVMe  
- checksummed  
- used **locally only** during training

No remote reads during the full run.

---

### **2.2 Target Steady-State Utilization**
**Primary training objective:**  
> Keep GPUs between **80%–90% utilization** in steady state.

Sub-objectives:
- eliminate dataloader stalls  
- minimize synchronization points  
- reduce per-step variance  
- maintain sustained ~2.1M tokens/step

---

### **2.3 Control Global Batch via Grad Accumulation**
Don't chase large microbatches early.  
Use:
- microbatch = 8–12  
- global batch = 4096  
- grad accumulation = 16 (adjustable)

Adjust microbatch only if:
- memory is underutilized  
- activations checkpointing is removed  
- stability is confirmed

---

### **2.4 FSDP as the Default Sharding Strategy**
FSDP full-shard is the default due to:
- predictable memory use  
- deterministic scaling  
- better BF16 stability  
- easier multi-node behavior

Activation checkpointing:
- **ON** during V1/V2  
- **OPTIONAL** during V3 and full runs (remove for more throughput)

---

### **2.5 Checkpoint Cadence**
Use a **4–6 hour checkpoint interval** to balance:
- safety  
- IO cost  
- recovery time  
- log compactness

Rotate last N=4–6 checkpoints  
Store in NVMe → Wasabi

---

## 3. IO Strategy

### **3.1 Deterministic Shards Only**
All data must:
- be preprocessed  
- be versioned  
- have known checksums  
- match the manifest

No on-the-fly shard generation.

---

### **3.2 Dataloader Saturation First**
Dataloader config must achieve:
- workers per GPU: 4–6  
- prefetch factor: 4–6  
- pinned memory: true  
- persistent workers: true

Run standalone dataloader stress tests before V2.

---

### **3.3 Per-Shard Validation**
Before training:
- verify schema  
- verify shape  
- verify sample count  
- verify hash  
- verify read speed

Training halts if any shard fails integrity.

---

## 4. Distributed Strategy

### **4.1 Single-Node First, Always**
Single-node 16×H100 runs form the foundation.  
Only move to multi-node once:
- NCCL is verified  
- FSDP is verified  
- IO is verified  
- V1/V2 are clean  

---

### **4.2 Multi-Node Constraints**
When using 2×8 or 4×8 layouts:
- ensure MASTER_ADDR stability  
- ensure IB bandwidth tests pass  
- set NCCL env vars:
  - `NCCL_ASYNC_ERROR_HANDLING=1`
  - `NCCL_BLOCKING_WAIT=0`
  - `TORCH_DISTRIBUTED_DEBUG=OFF`

Avoid:
- uneven shard splits  
- node-local NVMe imbalance  
- oversubscription of workers

---

### **4.3 No Divergent Configurations**
The config used in V3 must be the **exact** config used in the full run (minus step count and checkpoint cadence).

This prevents:
- version drift  
- seed mismatch  
- optimization instability  
- unexpected divergence

---

## 5. Risk Strategy

### **5.1 Fail Fast During Pre-Flight**
If something breaks in V1 or V2:
- do NOT “patch”  
- do NOT continue  
- fix root cause  
- rerun the stage

A failed pre-flight stage means the full run is unsafe to launch.

---

### **5.2 Strict Loss Monitoring**
Enable:
- `stop_on_nan_loss=true`  
- gradient clipping = 1.0  
- periodic evaluation every 10k–20k steps

Early signs of divergence must abort the run immediately.

---

### **5.3 No Mid-Run Parameter Changes**
Once the full run begins:
- no layer count changes  
- no optimizer changes  
- no LR schedule changes  
- no batch size changes  
- no seed changes  
- no version jumps

Any change = restart the run with a new `run_id`.

---

## 6. Throughput Optimization Strategy

### **6.1 Focus on Bottlenecks in Order**
1. **IO stalls**  
2. **Synchronization points**  
3. **Microbatch tuning**  
4. **Activation checkpointing removal**  
5. **Prefetch and workers**  
6. **NCCL environment**  

The order matters — fix upstream bottlenecks first.

---

### **6.2 Remove Checkpoint Overhead**
During steady-state profiling:
- run ~6 hours with checkpointing disabled  
- measure raw tokens/sec  
- reintroduce checkpoints after tuning

---

### **6.3 Checkpoint on Separate Thread**
Use asynchronous IO to:
- write checkpoints  
- flush logs  
- push to Wasabi  

Without blocking forward/backward passes.

---

## 7. Validation Strategy

### **7.1 End-to-End Dry Runs**
Before launching the full run:
- simulate 1 batch of forward/backward  
- simulate 1 checkpoint write  
- simulate 1 eval cycle  
- simulate shard loading  
- measure full throughput path

Everything must work on the **exact** final configuration.

---

### **7.2 Golden Step Consistency**
During V3, pick several “golden steps”:
- step 1000  
- step 5000  
- step 10000  

Record:
- tokens/sec  
- memory usage  
- throughput variance  
- CPU IO load

Your full run must hit within ±10% of those numbers.

---

## 8. Recovery Strategy

### **8.1 Mid-Run Resume**
If crash:
- resume from last checkpoint  
- re-validate shards  
- run dry batch  
- continue

---

### **8.2 Hard Reset**
If divergence or corruption:
- terminate run  
- revert to stable checkpoint  
- diagnose  
- relaunch under new `run_id`  
- never overwrite previous logs

---

## 9. Completion Strategy

### **9.1 End of Run**
After ~400k–450k steps:
- select final checkpoint  
- freeze model artifacts  
- archive training logs  
- register run in manifest  
- tag: `nexa_mol_fm_v1`

---

# **Summary**
This compute strategy ensures:
- correctness before speed  
- stability before throughput  
- risk reduction before optimization  
- deterministic behavior  
- safe scaling across 16–32 H100s  
- maximum training reliability  
- minimal wasted compute  
- consistent results across weeks of training

**This is the doctrine.  
This is how Atheron Labs runs compute.**
