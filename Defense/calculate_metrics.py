import pandas as pd
import numpy as np

INPUT_FILE = "generated new result file directory"

df = pd.read_excel(INPUT_FILE)
total_rows = len(df)
total_responses = total_rows * 2

print(f"Loaded {total_rows} rows")
print(f"Total Responses: {total_responses} (Turn2 + Turn3)")

# ============= REFUSAL RATE =============
turn2_blocked = df['Turn2_IsHarmful'].sum()
turn3_blocked = df['Turn3_IsHarmful'].sum()
total_blocked = turn2_blocked + turn3_blocked

turn2_refusal_rate = (turn2_blocked / total_rows) * 100
turn3_refusal_rate = (turn3_blocked / total_rows) * 100
overall_refusal_rate = (total_blocked / total_responses) * 100

print(f"Turn 2:")
print(f"   Blocked: {turn2_blocked}/{total_rows} ({turn2_refusal_rate:.2f}%)")
print(f"   Allowed: {total_rows - turn2_blocked}/{total_rows} ({100 - turn2_refusal_rate:.2f}%)")
print()
print(f"Turn 3:")
print(f"   Blocked: {turn3_blocked}/{total_rows} ({turn3_refusal_rate:.2f}%)")
print(f"   Allowed: {total_rows - turn3_blocked}/{total_rows} ({100 - turn3_refusal_rate:.2f}%)")
print(f"Overall Refusal Rate: {total_blocked}/{total_responses} ({overall_refusal_rate:.2f}%)")

# ============= ASR (Attack Success Rate) =============
print("ASR (Attack Success Rate) - Jailbreak Success")

# ASR for ORIGINAL responses (before defense)
turn2_original_jailbroken = df['Turn2_Original_IsJailbroken'].sum()
turn3_original_jailbroken = df['Turn3_Original_IsJailbroken'].sum()
total_original_jailbroken = turn2_original_jailbroken + turn3_original_jailbroken

turn2_original_asr = (turn2_original_jailbroken / total_rows) * 100
turn3_original_asr = (turn3_original_jailbroken / total_rows) * 100
overall_original_asr = (total_original_jailbroken / total_responses) * 100

print(f"ORIGINAL (Before Defense):")
print(f"   Turn 2 ASR: {turn2_original_jailbroken}/{total_rows} ({turn2_original_asr:.2f}%)")
print(f"   Turn 3 ASR: {turn3_original_jailbroken}/{total_rows} ({turn3_original_asr:.2f}%)")
print(f"   Overall ASR: {total_original_jailbroken}/{total_responses} ({overall_original_asr:.2f}%)")

# ASR for DEFENDED responses (after defense)
turn2_defended_jailbroken = df['Turn2_Defended_IsJailbroken'].sum()
turn3_defended_jailbroken = df['Turn3_Defended_IsJailbroken'].sum()
total_defended_jailbroken = turn2_defended_jailbroken + turn3_defended_jailbroken

turn2_defended_asr = (turn2_defended_jailbroken / total_rows) * 100
turn3_defended_asr = (turn3_defended_jailbroken / total_rows) * 100
overall_defended_asr = (total_defended_jailbroken / total_responses) * 100

print(f"DEFENDED (After Defense):")
print(f"   Turn 2 ASR: {turn2_defended_jailbroken}/{total_rows} ({turn2_defended_asr:.2f}%)")
print(f"   Turn 3 ASR: {turn3_defended_jailbroken}/{total_rows} ({turn3_defended_asr:.2f}%)")
print(f"   Overall ASR: {total_defended_jailbroken}/{total_responses} ({overall_defended_asr:.2f}%)")

# ASR Reduction
asr_reduction = overall_original_asr - overall_defended_asr
asr_reduction_percent = (asr_reduction / overall_original_asr * 100) if overall_original_asr > 0 else 0

print(f"ASR Reduction: {asr_reduction:.2f} percentage points")
print(f"Relative Improvement: {asr_reduction_percent:.2f}% reduction in attack success")


# ============= SUMMARY TABLE =============
print("SUMMARY COMPARISON TABLE")
summary_data = {
    'Metric': [
        'Total Responses',
        'Refusal Rate (%)',
        'ASR - Original (%)',
        'ASR - Defended (%)',
        'ASR Reduction (pp)',
    ],
    'Value': [
        total_responses,
        f"{overall_refusal_rate:.2f}",
        f"{overall_original_asr:.2f}",
        f"{overall_defended_asr:.2f}",
        f"{asr_reduction:.2f}",
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("ANALYSIS COMPLETE!")