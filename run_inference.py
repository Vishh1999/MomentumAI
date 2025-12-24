import pandas as pd
import pickle
from openai import OpenAI
import textwrap

# Load trained models
with open("model_pkl_files/lr_model_pb.pkl", "rb") as f:
    pb_model = pickle.load(f)

with open("model_pkl_files/lr_model_streak.pkl", "rb") as f:
    streak_model = pickle.load(f)

FEATURES = [
    "training_load",
    "average_heartrate",
    "moving_time",
    "streak_length"
]
# In a real system, the following inputs would be
# collected automatically from workout logs, wearables,
# or backend services. For demo purposes, we simulate this
# with manual user input.

print("\nEnter current user training details:\n")

training_load = float(input("Training load: "))
avg_hr = float(input("Average heart rate: "))
moving_time = float(input("Moving time (minutes): "))
streak_length = int(input("Current streak length: "))

# Build feature vector
X_user = pd.DataFrame([{
    "training_load": training_load,
    "average_heartrate": avg_hr,
    "moving_time": moving_time,
    "streak_length": streak_length
}])

# Model inference
pb_prob = pb_model.predict_proba(X_user.values)[0, 1]
streak_risk = streak_model.predict_proba(X_user.values)[0, 1]

# Product signal logic
def derive_signal(pb, streak):
    if pb >= 0.65 and streak < 0.4:
        return "PB_OPPORTUNITY"
    elif streak >= 0.6:
        return "STREAK_RISK"
    else:
        return "MAINTAIN"

signal = derive_signal(pb_prob, streak_risk)

print(textwrap.fill(f"For training load, {training_load}, a streak length of {streak_length}, "
      f"average heartrate of {avg_hr} and a moving time of {moving_time}, "
      f"the personal best probability is {pb_prob} and streak risk is {streak_risk}.", width=60))
print(f"This produces a signal {signal} to the LLM!")


# LLM prompt (MomentumAI)
prompt = f"""
You are MomentumAI, a calm and supportive training coach inside a hybrid fitness app.
This message is shown immediately after the user finishes a workout.

User context:
- Recent training load from this session: {training_load}
- Current workout streak: {streak_length} sessions
- Coaching signal: {signal}

Guidelines:
- Write 2–3 short sentences.
- Acknowledge that the workout has just finished.
- Sound like an experienced coach, not a hype influencer.
- Be specific and grounded in the session that just happened.
- Match the message to the coaching signal:
  * PB_OPPORTUNITY -- reinforce quality effort and readiness, without guaranteeing outcomes.
  * STREAK_RISK -- acknowledge effort and gently suggest recovery or restraint next.
  * MAINTAIN -- reinforce consistency and steady progress.
- Focus on reflection, recovery, or the *next sensible step*.
- Do NOT mention data, scores, probabilities, or machine learning.
- Do NOT exaggerate or promise results.

Write the coaching message now.
"""

# LLM call, using OpenAI for this project can swap this with any other LLM
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=120
)
LLM_response = response.choices[0].message.content
print("\nCoaching Insight from MomentumAI:\n")
print(textwrap.fill(LLM_response.strip(), width=60))
