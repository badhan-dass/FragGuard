import pandas as pd
import numpy as np

# ============= CONFIGURATION =============
# Change this to your actual file path
INPUT_FILE = "generated new result file directory"

print("="*80)
print("📊 CALCULATING METRICS FROM DEFENDED OUTPUT")
print("="*80)
print(f"📁 Input File: {INPUT_FILE}")
print("="*80)

# Load the data
df = pd.read_excel(INPUT_FILE)
total_rows = len(df)
total_responses = total_rows * 2  # Turn2 + Turn3

print(f"\n✅ Loaded {total_rows} rows")
print(f"📝 Total Responses: {total_responses} (Turn2 + Turn3)")

# ============= REFUSAL RATE =============
print("\n" + "="*80)
print("🚨 REFUSAL RATE (Defense Blocking)")
print("="*80)

# Count how many were blocked by defense
turn2_blocked = df['IsHarmful'].sum()
# turn3_blocked = df['Turn3_IsHarmful'].sum()
total_blocked = turn2_blocked #+ turn3_blocked

turn2_refusal_rate = (turn2_blocked / total_rows) * 100
# turn3_refusal_rate = (turn3_blocked / total_rows) * 100
overall_refusal_rate = (total_blocked / total_responses) * 100

print(f"🔵 Single Turn:")
print(f"   Blocked: {turn2_blocked}/{total_rows} ({turn2_refusal_rate:.2f}%)")
print(f"   Allowed: {total_rows - turn2_blocked}/{total_rows} ({100 - turn2_refusal_rate:.2f}%)")
print()
# print(f"🟢 Turn 3:")
# print(f"   Blocked: {turn3_blocked}/{total_rows} ({turn3_refusal_rate:.2f}%)")
# print(f"   Allowed: {total_rows - turn3_blocked}/{total_rows} ({100 - turn3_refusal_rate:.2f}%)")
# print()
# print(f"🎯 Overall Refusal Rate: {total_blocked}/{total_responses} ({overall_refusal_rate:.2f}%)")

# ============= ASR (Attack Success Rate) =============
print("\n" + "="*80)
print("⚔️ ASR (Attack Success Rate) - Jailbreak Success")
print("="*80)

# ASR for ORIGINAL responses (before defense)
turn2_original_jailbroken = df['Original_IsJailbroken'].sum()
# turn3_original_jailbroken = df['Turn3_Original_IsJailbroken'].sum()
total_original_jailbroken = turn2_original_jailbroken #+ turn3_original_jailbroken

turn2_original_asr = (turn2_original_jailbroken / total_rows) * 100
# turn3_original_asr = (turn3_original_jailbroken / total_rows) * 100
overall_original_asr = (total_original_jailbroken / total_responses) * 100

print(f"📝 ORIGINAL (Before Defense):")
print(f"   Single Turn ASR: {turn2_original_jailbroken}/{total_rows} ({turn2_original_asr:.2f}%)")
# print(f"   Turn 3 ASR: {turn3_original_jailbroken}/{total_rows} ({turn3_original_asr:.2f}%)")
print(f"   Overall ASR: {total_original_jailbroken}/{total_responses} ({overall_original_asr:.2f}%)")
print()

# ASR for DEFENDED responses (after defense)
turn2_defended_jailbroken = df['Defended_IsJailbroken'].sum()
# turn3_defended_jailbroken = df['Turn3_Defended_IsJailbroken'].sum()
total_defended_jailbroken = turn2_defended_jailbroken #+ turn3_defended_jailbroken

turn2_defended_asr = (turn2_defended_jailbroken / total_rows) * 100
# turn3_defended_asr = (turn3_defended_jailbroken / total_rows) * 100
overall_defended_asr = (total_defended_jailbroken / total_responses) * 100

print(f"🛡️ DEFENDED (After Defense):")
print(f"   Single Turn ASR: {turn2_defended_jailbroken}/{total_rows} ({turn2_defended_asr:.2f}%)")
# print(f"   Turn 3 ASR: {turn3_defended_jailbroken}/{total_rows} ({turn3_defended_asr:.2f}%)")
print(f"   Overall ASR: {total_defended_jailbroken}/{total_responses} ({overall_defended_asr:.2f}%)")
print()

# ASR Reduction
asr_reduction = overall_original_asr - overall_defended_asr
asr_reduction_percent = (asr_reduction / overall_original_asr * 100) if overall_original_asr > 0 else 0

print(f"✨ ASR Reduction: {asr_reduction:.2f} percentage points")
print(f"✨ Relative Improvement: {asr_reduction_percent:.2f}% reduction in attack success")

# ============= AVERAGE TOXICITY SCORE =============
print("\n" + "="*80)
print("🧪 AVERAGE TOXICITY SCORE (1-5 scale)")
print("="*80)

# ORIGINAL responses
turn2_original_avg = df['Original_EvalScore'].mean()
# turn3_original_avg = df['Turn3_Original_EvalScore'].mean()
overall_original_avg = (df['Original_EvalScore'].sum() / total_responses)

turn2_original_std = df['Original_EvalScore'].std()
# turn3_original_std = df['Turn3_Original_EvalScore'].std()
overall_original_std = pd.concat([df['Original_EvalScore']]).std()

print(f"📝 ORIGINAL (Before Defense):")
print(f"   Single Turn: {turn2_original_avg:.2f} ± {turn2_original_std:.2f}")
# print(f"   Turn 3: {turn3_original_avg:.2f} ± {turn3_original_std:.2f}")
print(f"   Overall: {overall_original_avg:.2f} ± {overall_original_std:.2f}")
print()

# DEFENDED responses
turn2_defended_avg = df['Defended_EvalScore'].mean()
# turn3_defended_avg = df['Turn3_Defended_EvalScore'].mean()
overall_defended_avg = (df['Defended_EvalScore'].sum() / total_responses)

turn2_defended_std = df['Defended_EvalScore'].std()
# turn3_defended_std = df['Turn3_Defended_EvalScore'].std()
overall_defended_std = pd.concat([df['Defended_EvalScore']]).std()

print(f"🛡️ DEFENDED (After Defense):")
print(f"   Single Turn: {turn2_defended_avg:.2f} ± {turn2_defended_std:.2f}")
# print(f"   Turn 3: {turn3_defended_avg:.2f} ± {turn3_defended_std:.2f}")
print(f"   Overall: {overall_defended_avg:.2f} ± {overall_defended_std:.2f}")
print()

# Toxicity Score Reduction
toxicity_reduction = overall_original_avg - overall_defended_avg
toxicity_reduction_percent = (toxicity_reduction / overall_original_avg * 100) if overall_original_avg > 0 else 0

print(f"✨ Toxicity Reduction: {toxicity_reduction:.2f} points")
print(f"✨ Relative Improvement: {toxicity_reduction_percent:.2f}% reduction in toxicity")

# ============= DISTRIBUTION ANALYSIS =============
print("\n" + "="*80)
print("📊 TOXICITY SCORE DISTRIBUTION")
print("="*80)

print("📝 ORIGINAL:")
for score in range(1, 6):
    turn2_count = (df['Original_EvalScore'] == score).sum()
    # turn3_count = (df['Turn3_Original_EvalScore'] == score).sum()
    total_count = turn2_count # + turn3_count
    percentage = (total_count / total_responses) * 100
    print(f"   Score {score}: {total_count}/{total_responses} ({percentage:.1f}%)")

print()
print("🛡️ DEFENDED:")
for score in range(1, 6):
    turn2_count = (df['Defended_EvalScore'] == score).sum()
    # turn3_count = (df['Turn3_Defended_EvalScore'] == score).sum()
    total_count = turn2_count
    percentage = (total_count / total_responses) * 100
    print(f"   Score {score}: {total_count}/{total_responses} ({percentage:.1f}%)")

# ============= SUMMARY TABLE =============
print("\n" + "="*80)
print("📋 SUMMARY COMPARISON TABLE")
print("="*80)

summary_data = {
    'Metric': [
        'Total Responses',
        'Refusal Rate (%)',
        'ASR - Original (%)',
        'ASR - Defended (%)',
        'ASR Reduction (pp)',
        'Avg Toxicity - Original',
        'Avg Toxicity - Defended',
        'Toxicity Reduction'
    ],
    'Value': [
        total_responses,
        f"{overall_refusal_rate:.2f}",
        f"{overall_original_asr:.2f}",
        f"{overall_defended_asr:.2f}",
        f"{asr_reduction:.2f}",
        f"{overall_original_avg:.2f}",
        f"{overall_defended_avg:.2f}",
        f"{toxicity_reduction:.2f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# ============= SAVE METRICS TO FILE (OPTIONAL) =============
print("\n" + "="*80)
print("💾 SAVING METRICS SUMMARY")
print("="*80)

metrics_output = INPUT_FILE.replace('.xlsx', '_METRICS_SUMMARY.txt')
with open(metrics_output, 'w') as f:
    f.write("="*80 + "\n")
    f.write("DEFENSE METRICS SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Input File: {INPUT_FILE}\n")
    f.write(f"Total Rows: {total_rows}\n")
    f.write(f"Total Responses: {total_responses}\n\n")
    
    f.write("REFUSAL RATE:\n")
    f.write(f"  Overall: {overall_refusal_rate:.2f}%\n")
    f.write(f"  Turn 2: {turn2_refusal_rate:.2f}%\n")
    # f.write(f"  Turn 3: {turn3_refusal_rate:.2f}%\n\n")
    
    f.write("ASR (Attack Success Rate):\n")
    f.write(f"  Original Overall: {overall_original_asr:.2f}%\n")
    f.write(f"  Defended Overall: {overall_defended_asr:.2f}%\n")
    f.write(f"  ASR Reduction: {asr_reduction:.2f} pp ({asr_reduction_percent:.2f}% relative)\n\n")
    
    f.write("AVERAGE TOXICITY SCORE (1-5):\n")
    f.write(f"  Original Overall: {overall_original_avg:.2f} ± {overall_original_std:.2f}\n")
    f.write(f"  Defended Overall: {overall_defended_avg:.2f} ± {overall_defended_std:.2f}\n")
    f.write(f"  Toxicity Reduction: {toxicity_reduction:.2f} points ({toxicity_reduction_percent:.2f}% relative)\n\n")
    
    f.write("="*80 + "\n")

print(f"✅ Metrics saved to: {metrics_output}")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)