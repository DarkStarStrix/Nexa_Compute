#!/usr/bin/env python3
"""Calculate training costs for distilling Falcon3-10B-Base."""

# GPU Pricing per hour
RTX_5090_PRICE = 0.92  # $0.92/hr
A100_PRICE = 1.00  # $1.00/hr
H100_PRICE = 2.00  # $2.00/hr

# Model specs (Falcon3-10B-Base)
MODEL_SIZE_GB = 20  # BF16 weights: 10B params × 2 bytes = 20GB
MODEL_PARAMS = 10_000_000_000

# Training assumptions
# Student model: 1B-3B parameters (typical for distillation)
STUDENT_SIZES = {
    "1B": {"params": 1_000_000_000, "size_gb": 2, "name": "1B Student"},
    "3B": {"params": 3_000_000_000, "size_gb": 6, "name": "3B Student"},
    "7B": {"params": 7_000_000_000, "size_gb": 14, "name": "7B Student"},
}

# Dataset size (from our analysis)
DATASET_SIZE = 125_000  # rows
TOKENS_PER_SAMPLE = 400  # Average tokens per sample
TOTAL_TOKENS = DATASET_SIZE * TOKENS_PER_SAMPLE  # 50M tokens

# Training hyperparameters
EPOCHS = 3  # Typical for distillation
BATCH_SIZE_PER_GPU = {
    "RTX_5090": {"1B": 8, "3B": 4, "7B": 2},  # 24GB VRAM
    "A100": {"1B": 32, "3B": 16, "7B": 8},    # 40GB VRAM
    "H100": {"1B": 64, "3B": 32, "7B": 16},   # 80GB VRAM
}

GRADIENT_ACCUMULATION = 4  # Effective batch size multiplier

# Training speed estimates (tokens/second)
# Conservative estimates based on model size and GPU
TRAINING_SPEED = {
    "RTX_5090": {"1B": 2000, "3B": 800, "7B": 300},   # tokens/sec
    "A100": {"1B": 5000, "3B": 2000, "7B": 1000},
    "H100": {"1B": 10000, "3B": 5000, "7B": 2500},
}


def calculate_training_time(gpu_type: str, student_size: str) -> float:
    """Calculate training time in hours."""
    tokens_per_epoch = TOTAL_TOKENS
    total_tokens = tokens_per_epoch * EPOCHS
    
    speed = TRAINING_SPEED[gpu_type][student_size]
    time_seconds = total_tokens / speed
    time_hours = time_seconds / 3600
    
    return time_hours


def calculate_cost(gpu_type: str, student_size: str) -> dict:
    """Calculate total training cost."""
    gpu_price = {
        "RTX_5090": RTX_5090_PRICE,
        "A100": A100_PRICE,
        "H100": H100_PRICE,
    }[gpu_type]
    
    training_time = calculate_training_time(gpu_type, student_size)
    total_cost = training_time * gpu_price
    
    # Add 20% buffer for setup, debugging, evaluation
    total_cost_with_buffer = total_cost * 1.2
    
    return {
        "gpu_type": gpu_type,
        "student_size": student_size,
        "training_hours": training_time,
        "cost_per_hour": gpu_price,
        "base_cost": total_cost,
        "total_cost_with_buffer": total_cost_with_buffer,
        "tokens_per_sec": TRAINING_SPEED[gpu_type][student_size],
        "batch_size": BATCH_SIZE_PER_GPU[gpu_type][student_size],
    }


def main():
    print("=" * 80)
    print("FALCON3-10B-BASE DISTILLATION TRAINING COST ANALYSIS")
    print("=" * 80)
    print(f"\nTeacher Model: Falcon3-10B-Base (10B parameters, 20GB BF16)")
    print(f"Dataset: {DATASET_SIZE:,} samples, {TOTAL_TOKENS/1_000_000:.1f}M tokens total")
    print(f"Training: {EPOCHS} epochs")
    print(f"\nGPU Pricing:")
    print(f"  RTX 5090: ${RTX_5090_PRICE}/hr")
    print(f"  A100: ${A100_PRICE}/hr")
    print(f"  H100: ${H100_PRICE}/hr")
    
    print("\n" + "=" * 80)
    print("COST BREAKDOWN BY GPU AND STUDENT SIZE")
    print("=" * 80)
    
    results = []
    for gpu in ["RTX_5090", "A100", "H100"]:
        for size in ["1B", "3B", "7B"]:
            result = calculate_cost(gpu, size)
            results.append(result)
            
            print(f"\n{gpu} + {STUDENT_SIZES[size]['name']} Student:")
            print(f"  Training time: {result['training_hours']:.1f} hours")
            print(f"  Base cost: ${result['base_cost']:.2f}")
            print(f"  Total cost (with 20% buffer): ${result['total_cost_with_buffer']:.2f}")
            print(f"  Speed: {result['tokens_per_sec']:,} tokens/sec")
            print(f"  Batch size: {result['batch_size']}")
    
    # Find best options
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Best cost option
    best_cost = min(results, key=lambda x: x['total_cost_with_buffer'])
    print(f"\n💰 Best Cost Option:")
    print(f"  {best_cost['gpu_type']} + {STUDENT_SIZES[best_cost['student_size']]['name']} Student")
    print(f"  Cost: ${best_cost['total_cost_with_buffer']:.2f}")
    print(f"  Time: {best_cost['training_hours']:.1f} hours")
    
    # Best speed option
    best_speed = min(results, key=lambda x: x['training_hours'])
    print(f"\n⚡ Fastest Option:")
    print(f"  {best_speed['gpu_type']} + {STUDENT_SIZES[best_speed['student_size']]['name']} Student")
    print(f"  Cost: ${best_speed['total_cost_with_buffer']:.2f}")
    print(f"  Time: {best_speed['training_hours']:.1f} hours")
    
    # ROI analysis
    print("\n" + "=" * 80)
    print("ROI & VALUE ANALYSIS")
    print("=" * 80)
    
    # Portfolio value
    print("\n📊 Portfolio Value:")
    print("  ✅ Advanced ML expertise demonstration")
    print("  ✅ Knowledge distillation implementation")
    print("  ✅ Large model fine-tuning experience")
    print("  ✅ Production-ready model deployment")
    print("  ✅ Cost optimization skills")
    print("  ✅ Open-source contribution potential")
    
    # Monetization potential
    print("\n💰 Monetization Potential:")
    print("  1. API Service:")
    print("     - Deploy distilled model as API")
    print("     - Charge per inference: $0.001-0.01 per request")
    print("     - Break-even: ~100k requests (at $0.005/req)")
    
    print("\n  2. Model Licensing:")
    print("     - License distilled model to businesses")
    print("     - One-time: $5k-50k depending on use case")
    print("     - Subscription: $500-2k/month")
    
    print("\n  3. Consulting/Expertise:")
    print("     - Offer distillation services")
    print("     - Typical project: $10k-50k")
    print("     - Demonstrates expertise from portfolio")
    
    print("\n  4. Open Source Contribution:")
    print("     - Release model on HuggingFace")
    print("     - Build reputation in ML community")
    print("     - Leads to job opportunities, consulting")
    
    # Cost comparison
    print("\n" + "=" * 80)
    print("COST COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'GPU':<12} {'Student Size':<15} {'Time (hrs)':<12} {'Cost ($)':<12}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x['total_cost_with_buffer']):
        print(f"{r['gpu_type']:<12} {STUDENT_SIZES[r['student_size']]['name']:<15} "
              f"{r['training_hours']:<12.1f} ${r['total_cost_with_buffer']:<12.2f}")
    
    # Recommendation
    print("\n" + "=" * 80)
    print("🎯 RECOMMENDATION")
    print("=" * 80)
    print(f"\nFor fastest ROI: Use {best_cost['gpu_type']} with 1B student")
    print(f"  - Cost: ${best_cost['total_cost_with_buffer']:.2f}")
    print(f"  - Time: {best_cost['training_hours']:.1f} hours")
    print(f"  - Best balance of cost and speed")
    print(f"\nFor best quality: Use H100 with 3B-7B student")
    print(f"  - Higher cost but better model performance")
    print(f"  - Better for portfolio demonstration")
    
    print(f"\n💡 Total Investment Breakdown:")
    teacher_cost = 100  # GPT-mini teacher generation
    training_cost = best_cost['total_cost_with_buffer']
    total = teacher_cost + training_cost
    print(f"  Teacher generation (GPT-mini): ${teacher_cost:.2f}")
    print(f"  Training cost: ${training_cost:.2f}")
    print(f"  Total: ${total:.2f}")
    print(f"\n  ROI Break-even: {total/0.005:.0f} API calls at $0.005/request")
    print(f"  ROI Break-even: {total/1000:.0f} months at $1k/month licensing")


if __name__ == "__main__":
    main()

