import pandas as pd
import streamlit as st

import ui
from core import expert_system as es

ui.setup("Expert System", "🩺", "A rule-based symptom checker. Educational demo only, not medical advice.")

if "custom_rules" not in st.session_state:
    st.session_state["custom_rules"] = []

rules = list(st.session_state["custom_rules"]) + es.DEFAULT_RULES

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Symptoms")
    boxes = st.columns(2)
    selected = [
        key for i, (key, label) in enumerate(es.SYMPTOMS.items())
        if boxes[i % 2].checkbox(label, key=f"sym_{key}")
    ]

diagnosis = es.diagnose(selected, rules)

with right:
    st.subheader("Result")
    if not selected:
        st.info("Select at least one symptom.")
    elif diagnosis.primary is None:
        st.info("No clear diagnosis from the current rule base.")
    else:
        rule = diagnosis.primary
        (st.error if rule.urgent else st.success)(f"**{rule.conclusion}**  \nRule {rule.id}: IF {rule.describe()}")
        if rule.advice:
            st.write(rule.advice)
        others = [r for r in diagnosis.fired if r is not rule]
        if others:
            st.caption("Also matched (lower priority): " + ", ".join(r.conclusion for r in others))

st.subheader("Reasoning trace")
st.caption("Rules are checked from the top; the first rule that fires becomes the primary result.")
st.dataframe(
    pd.DataFrame([
        {
            "Priority": i + 1,
            "Rule": r.id,
            "IF (all of)": r.describe(),
            "THEN": r.conclusion,
            "Fired": "✅" if ok else "-",
            "Used as result": "⭐" if diagnosis.primary is r else "",
        }
        for i, (r, ok) in enumerate(diagnosis.trace)
    ]),
    hide_index=True,
)

with st.expander("➕ Add your own rule"):
    with st.form("new_rule", clear_on_submit=True):
        conclusion = st.text_input("THEN conclusion", placeholder="e.g. Seasonal allergy")
        conditions = st.multiselect("IF all of these symptoms are present", list(es.SYMPTOMS), format_func=es.SYMPTOMS.get)
        if st.form_submit_button("Add rule with highest priority"):
            if conclusion.strip() and conditions:
                n = len(st.session_state["custom_rules"]) + 1
                st.session_state["custom_rules"].insert(0, es.Rule(f"C{n}", conclusion.strip(), frozenset(conditions)))
                st.rerun()
            else:
                st.warning("Give the rule a conclusion and at least one symptom.")
    if st.session_state["custom_rules"] and st.button("Remove my custom rules"):
        st.session_state["custom_rules"] = []
        st.rerun()

ui.show_originals("03_Rule_Based_Expert_System.py")
