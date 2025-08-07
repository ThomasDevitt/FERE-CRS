
import os
import json
import csv
import time
import random
from typing import List, Dict, Any, Tuple, Optional
import ast

# --- Required Libraries ---
# Please install these libraries before running:
# pip install pandas google-generativeai matplotlib seaborn

import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global Configuration ---
# IMPORTANT: Set your Google API Key as an environment variable.
# The script will read the environment variable "GOOGLE_API_KEY".
API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")

# --- File-based "Cognitive Choreographer" Assets ---

RULES_TXT_CONTENT = """
# Component Types and Hierarchy
- The 'heater' is an 'electrical_component'.
- The 'pump' is an 'electrical_component'.
- The 'psu' (power supply unit) is an 'electrical_component'.
- The 'power_cord' is an 'electrical_component'.
- The 'grinder' is an 'electrical_component'.
- The 'control_board' is an 'electrical_component'.

# Causal Power Chain
- The 'heater' requires the 'psu' to have status 'ok'.
- The 'pump' requires the 'psu' to have status 'ok'.
- The 'grinder' requires the 'psu' to have status 'ok'.
- The 'psu' requires the 'power_cord' to have status 'ok'.
- The 'power_cord' requires an 'external_power_source' with status 'ok'.
- The 'control_board' requires the 'psu' to have status 'ok'.

# Symptom Causality
- A 'faulty' status for 'heater' can cause the 'cold_coffee' symptom.
- A 'faulty' status for 'psu' can cause the 'cold_coffee' symptom.
- A 'faulty' status for 'psu' can cause the 'no_power' symptom.
- A 'faulty' status for 'pump' can cause the 'no_water' symptom.
- A 'faulty' status for 'grinder' can cause the 'beans_not_ground' symptom.
- A 'faulty' status for 'control_board' can cause the 'no_power' symptom or 'erratic_behavior'.
"""

PROMPTS_JSON_CONTENT = """
{
  "UE_PARSE_SYMPTOM": "You are a helpful assistant. Parse the following user statement: '{user_statement}' into a single structured `[subject, predicate, object]` proposition. Example: 'The coffee is cold' becomes ['coffee_maker', 'has_symptom', 'cold_coffee']. Respond with the JSON list only.",
  
  "TE_GENERATE_HYPOTHESES": "You are a diagnostic reasoner. Given these established facts: {context}. And these inviolable rules: {rules}. What are the three most likely root cause components that could be faulty? List only the component names, separated by commas.",
  
  "CRS_EVAL_COHERENCE": "You are a logic evaluation engine. Given these inviolable rules: {rules}. Is the following hypothesis logically possible as a cause for the symptom '{symptom}'? Hypothesis: The '{hypothesis_component}' is faulty. Answer only with YES or NO.",
  
  "CRS_EVAL_PRAGMATIC_VALUE": "You are an evaluation engine. My goal is to 'find the single faulty component'. My current list of possible candidates has {candidate_count} items. If I perform the action 'test component {action_component}', and it turns out to be 'ok', I can remove it from my list. On a scale of 0.0 to 1.0, what is the pragmatic value of this action for achieving my goal? Provide only a single floating-point number.",

  "CRS_EVAL_INFO_VALUE": "You are an evaluation engine. My current list of possible faulty components is: {candidate_list}. On a scale of 0.0 to 1.0, how much useful information would the action 'test component {action_component}' provide for reducing my uncertainty? Provide only a single floating-point number.",

  "RAG_BASELINE_PROMPT": "You are an expert diagnostic agent. Your goal is to identify the single faulty component in a coffee maker. Here is the complete schematic and rules for the device: {rules}. The observed symptom is: {symptom}. Your diagnostic history, including test results, is: {history}. Based on all this information, what is the single best next action to take? Choose one from: 'test [component_name]' or 'declare [component_name] as faulty'. Respond with the action only."
}
"""

CONFIG_JSON_CONTENT = """
{
  "LLM_MODEL": "gemini-1.5-pro-latest",
  "CRS_WEIGHTS": {
    "R": 1.0,
    "P": 1.5,
    "I": 1.2,
    "C": 0.1
  },
  "ACTION_COSTS": {
    "test_component": 1.0,
    "ask_llm_for_hypotheses": 5.0,
    "declare_fault": 0.1
  },
  "MAX_STEPS_PER_TRIAL": 15
}
"""

SCENARIOS_JSON_CONTENT = """
[
    {"trial_id": 1, "faulty_component": "heater", "initial_symptom": "The coffee is cold but the machine has power."},
    {"trial_id": 2, "faulty_component": "pump", "initial_symptom": "The machine makes noise but no water comes out."},
    {"trial_id": 3, "faulty_component": "grinder", "initial_symptom": "The machine attempts to brew but the beans are not ground."},
    {"trial_id": 4, "faulty_component": "psu", "initial_symptom": "The machine shows no signs of power."},
    {"trial_id": 5, "faulty_component": "control_board", "initial_symptom": "The machine's lights are flickering and it behaves erratically."}
]
"""

# --- LLM Service Wrapper ---

class LLMService:
    """A simple wrapper for the Gemini API to handle retries and configuration."""
    def __init__(self, api_key: str, model_name: str):
        if api_key == "YOUR_API_KEY_HERE":
            raise ValueError("API Key not configured. Please set your GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"LLM Service Initialized with model: {model_name}")

    def query(self, prompt: str, retry_count=3, delay=5) -> Optional[str]:
        """Sends a prompt to the LLM and handles basic retries."""
        for attempt in range(retry_count):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"LLM query failed (attempt {attempt + 1}/{retry_count}): {e}")
                time.sleep(delay)
        return None

# --- Agent Implementations ---

class FERE_Agent:
    """The FERE-CRS MVP Agent, with corrected logic and logging."""
    def __init__(self, config, prompts, rules, llm_service):
        self.config = config
        self.prompts = prompts
        self.rules = rules
        self.llm = llm_service
        self.wm = {}

    def calculate_crs(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Calculates the projected CRS and returns the score and its components."""
        scores = {'R': 1.0, 'P': 0.0, 'I': 0.0, 'C_cost': self.config['ACTION_COSTS'].get(action['name'], 1.0)}
        
        if action['name'] == 'test_component':
            if len(self.wm['hypotheses']) <= 1:
                 scores['I'], scores['P'] = 0.0, 0.0
            else:
                p_prompt = self.prompts['CRS_EVAL_PRAGMATIC_VALUE'].format(
                    candidate_count=len(self.wm['hypotheses']),
                    action_component=action['params']['component']
                )
                i_prompt = self.prompts['CRS_EVAL_INFO_VALUE'].format(
                    candidate_list=", ".join(self.wm['hypotheses']),
                    action_component=action['params']['component']
                )
                try:
                    p_response = self.llm.query(p_prompt)
                    i_response = self.llm.query(i_prompt)
                    scores['P'] = float(p_response) if p_response and p_response.replace('.', '', 1).isdigit() else 0.1
                    scores['I'] = float(i_response) if i_response and i_response.replace('.', '', 1).isdigit() else 0.1
                except (ValueError, TypeError):
                    scores['P'], scores['I'] = 0.1, 0.1
        
        elif action['name'] == 'ask_llm_for_hypotheses':
            scores['I'] = 0.9 if len(self.wm['hypotheses']) > 1 else 0.0
        
        elif action['name'] == 'declare_fault':
            scores['P'] = 2.0 if len(self.wm['hypotheses']) == 1 else 0.0
            scores['I'] = -0.5

        weights = self.config['CRS_WEIGHTS']
        total_score = (weights['R'] * scores['R']) + (weights['P'] * scores['P']) + (weights['I'] * scores['I']) - (weights['C'] * scores['C_cost'])
        
        return total_score, scores

    def run_diagnosis(self, trial_id: int, initial_symptom: str, faulty_component: str, logger: csv.DictWriter):
        self.wm = {
            'initial_symptom': initial_symptom,
            'hypotheses': ['heater', 'pump', 'grinder', 'psu', 'control_board'],
            'tested_ok': [],
            'history': []
        }
        
        for step in range(self.config['MAX_STEPS_PER_TRIAL']):
            print(f"\n--- FERE Agent Step {step+1} ---")
            print(f"Current hypotheses: {self.wm['hypotheses']}")

            candidate_actions = []
            if len(self.wm['hypotheses']) > 1:
                candidate_actions.append({'name': 'ask_llm_for_hypotheses', 'params': {}})
            
            for component in self.wm['hypotheses']:
                candidate_actions.append({'name': 'test_component', 'params': {'component': component}})
            
            if len(self.wm['hypotheses']) == 1:
                 candidate_actions.append({'name': 'declare_fault', 'params': {'component': self.wm['hypotheses'][0]}})

            if not candidate_actions: break

            print("Evaluating candidate actions...")
            scored_actions = []
            for action in candidate_actions:
                score, scores_dict = self.calculate_crs(action)
                print(f"  - Action: {action['name']}({action.get('params', {}).get('component', '')}), CRS: {score:.2f} (P:{scores_dict['P']:.2f}, I:{scores_dict['I']:.2f})")
                scored_actions.append(((score, scores_dict), action))
            
            (best_score, best_scores_dict), best_action = max(scored_actions, key=lambda item: item[0][0])
            
            action_text = f"Step {step+1}: Chose action '{best_action['name']}' with params {best_action['params']} [CRS: {best_score:.2f}]"
            print(action_text)
            self.wm['history'].append(action_text)
            self.log_step(trial_id, best_action, best_scores_dict, best_score, logger)

            if best_action['name'] == 'test_component':
                comp_to_test = best_action['params']['component']
                if comp_to_test == faulty_component:
                    print(f"  - Result: FAULTY. Hypothesis confirmed.")
                    self.wm['hypotheses'] = [comp_to_test]
                else:
                    print(f"  - Result: OK.")
                    if comp_to_test in self.wm['hypotheses']:
                        self.wm['hypotheses'].remove(comp_to_test)
                    self.wm['tested_ok'].append(comp_to_test)
            
            elif best_action['name'] == 'ask_llm_for_hypotheses':
                prompt = self.prompts['TE_GENERATE_HYPOTHESES'].format(context=json.dumps(self.wm), rules=self.rules)
                response = self.llm.query(prompt)
                if response:
                    new_hypotheses = [h.strip() for h in response.split(',')]
                    self.wm['hypotheses'] = [h for h in new_hypotheses if h in self.wm['hypotheses']]
                    print(f"  - New hypotheses: {self.wm['hypotheses']}")

            elif best_action['name'] == 'declare_fault':
                declared_comp = best_action['params']['component']
                success = (declared_comp == faulty_component)
                logger.writerow({'trial_id': trial_id, 'agent_type': 'FERE_CRS', 'outcome': 'success' if success else 'failure'})
                return

    def log_step(self, trial_id, action, scores, total_crs, logger):
        logger.writerow({
            'trial_id': trial_id,
            'agent_type': 'FERE_CRS',
            'step_number': len(self.wm['history']),
            'current_hypotheses_count': len(self.wm['hypotheses']),
            'chosen_action': action['name'],
            'action_params': json.dumps(action['params']),
            'R_score': scores['R'], 'P_score': scores['P'], 'I_score': scores['I'], 'C_score': scores['C_cost'], 
            'total_CRS': total_crs
        })

class RAG_Agent:
    """The RAG Baseline Agent."""
    def __init__(self, config, prompts, rules, llm_service):
        self.config = config
        self.prompts = prompts
        self.rules = rules
        self.llm = llm_service
        self.history = []

    def run_diagnosis(self, trial_id: int, initial_symptom: str, faulty_component: str, logger: csv.DictWriter):
        self.history = []
        
        for step in range(self.config['MAX_STEPS_PER_TRIAL']):
            prompt = self.prompts['RAG_BASELINE_PROMPT'].format(
                rules=self.rules,
                symptom=initial_symptom,
                history="\n".join(self.history)
            )
            response = self.llm.query(prompt)
            if not response: 
                logger.writerow({'trial_id': trial_id, 'agent_type': 'RAG_Baseline', 'outcome': 'failure'})
                return

            action_str = response.lower()
            
            feedback = ""
            if action_str.startswith('test'):
                try:
                    tested_comp = action_str.split('test ')[1].strip().replace("'", "").replace('"', '')
                    if tested_comp == faulty_component:
                        feedback = f"Result of testing {tested_comp}: FAULTY"
                    else:
                        feedback = f"Result of testing {tested_comp}: OK"
                except IndexError:
                    feedback = "Invalid test action format."
            
            self.history.append(f"Step {step+1}: Agent chose action '{action_str}'. {feedback}")
            self.log_step(trial_id, action_str, logger)

            if action_str.startswith('declare'):
                try:
                    declared_comp = action_str.replace('declare ', '').replace(' as faulty', '').strip().replace("'", "").replace('"', '')
                    success = (declared_comp == faulty_component)
                    logger.writerow({'trial_id': trial_id, 'agent_type': 'RAG_Baseline', 'outcome': 'success' if success else 'failure'})
                    return
                except:
                    logger.writerow({'trial_id': trial_id, 'agent_type': 'RAG_Baseline', 'outcome': 'failure'})
                    return
        
        logger.writerow({'trial_id': trial_id, 'agent_type': 'RAG_Baseline', 'outcome': 'failure'})

    def log_step(self, trial_id, action_str, logger):
        logger.writerow({
            'trial_id': trial_id,
            'agent_type': 'RAG_Baseline',
            'step_number': len(self.history)
        })

# --- Experiment Orchestration and Analysis ---

def setup_files():
    """Creates the choreographer's asset files if they don't exist."""
    if not os.path.exists('rules.txt'):
        with open('rules.txt', 'w') as f: f.write(RULES_TXT_CONTENT)
    if not os.path.exists('prompts.json'):
        with open('prompts.json', 'w') as f: f.write(PROMPTS_JSON_CONTENT)
    if not os.path.exists('config.json'):
        with open('config.json', 'w') as f: f.write(CONFIG_JSON_CONTENT)
    if not os.path.exists('scenarios.json'):
        with open('scenarios.json', 'w') as f: f.write(SCENARIOS_JSON_CONTENT)
    print("Asset files checked/created.")

def run_experiment(num_trials_per_scenario=100):
    """Main function to run the entire experiment and log results."""
    total_trials = num_trials_per_scenario * 5
    print(f"--- Starting Experiment ({total_trials} total trials) ---")
    
    with open('config.json', 'r') as f: config = json.load(f)
    with open('prompts.json', 'r') as f: prompts = json.load(f)
    with open('rules.txt', 'r') as f: rules = f.read()
    with open('scenarios.json', 'r') as f:
        base_scenarios = json.load(f)
        scenarios = [s for s in base_scenarios for _ in range(num_trials_per_scenario)]
        for i, s in enumerate(scenarios):
            s['trial_id'] = i + 1

    llm_service = LLMService(API_KEY, config['LLM_MODEL'])
    fere_agent = FERE_Agent(config, prompts, rules, llm_service)
    rag_agent = RAG_Agent(config, prompts, rules, llm_service)

    with open('results.csv', 'w', newline='') as csv_file:
        fieldnames = ['trial_id', 'agent_type', 'step_number', 'current_hypotheses_count', 
                      'chosen_action', 'action_params', 'R_score', 'P_score', 'I_score', 'C_score', 'total_CRS', 'outcome']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i, scenario in enumerate(scenarios):
            print(f"\n--- Running Trial {i+1}/{len(scenarios)} ---")
            print(f"Scenario: Faulty component is '{scenario['faulty_component']}'")

            print("Running FERE-CRS Agent...")
            fere_agent.run_diagnosis(scenario['trial_id'], scenario['initial_symptom'], scenario['faulty_component'], writer)

            print("Running RAG Baseline Agent...")
            rag_agent.run_diagnosis(scenario['trial_id'], scenario['initial_symptom'], scenario['faulty_component'], writer)

    print("\n--- Experiment Finished. Results logged to results.csv ---")

def analyze_results():
    """Analyzes the results.csv file and generates plots and stats."""
    print("\n--- Analyzing Results ---")
    try:
        df = pd.read_csv('results.csv')
    except FileNotFoundError:
        print("Error: results.csv not found. Please run the experiment first.")
        return

    # --- Data Reconstruction for Analysis ---
    print("Reconstructing summary data for analysis...")
    summary_data = []
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    rag_cost_per_step = config.get('ACTION_COSTS', {}).get('ask_llm_for_hypotheses', 5.0)

    for (trial_id, agent_type), data in df.groupby(['trial_id', 'agent_type']):
        outcome_row = data[data['outcome'].notna()]
        outcome = outcome_row['outcome'].iloc[0] if not outcome_row.empty else 'failure'
        
        total_cost = 0
        if agent_type == 'FERE_CRS':
            total_cost = data['C_score'].sum()
        else:
            total_cost = len(data[data['step_number'].notna()]) * rag_cost_per_step
        
        summary_data.append({'agent_type': agent_type, 'outcome': outcome, 'total_cost': total_cost})

    analysis_df = pd.DataFrame(summary_data)
    
    # H1: Accuracy
    if not analysis_df.empty:
        accuracy = analysis_df.groupby('agent_type')['outcome'].value_counts(normalize=True).unstack().fillna(0)
        print("\n--- H1: Accuracy ---")
        print(accuracy)
        
        plt.figure(figsize=(8, 6))
        if 'success' in accuracy.columns:
            accuracy.plot(kind='bar', y='success', ax=plt.gca(), rot=0, color=['#4C72B0', '#55A868'], legend=False)
        plt.title('H1: Success Rate by Agent Type')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1.05)
        plt.xlabel('Agent Type')
        plt.tight_layout()
        plt.savefig('H1_accuracy_chart.png')
        print("Saved H1_accuracy_chart.png")
        plt.close()
    else:
        print("Could not generate H1 chart: No outcome data found.")

    # H2: Efficiency
    efficiency_df = analysis_df[analysis_df['outcome'] == 'success']
    if not efficiency_df.empty:
        efficiency = efficiency_df.groupby('agent_type')['total_cost'].mean().reset_index()
        print("\n--- H2: Efficiency (Avg Cost per Success) ---")
        print(efficiency)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=efficiency, x='agent_type', y='total_cost', palette=['#4C72B0', '#55A868'])
        plt.title('H2: Average Cost per Successful Diagnosis')
        plt.ylabel('Average Total Cost')
        plt.xlabel('Agent Type')
        plt.tight_layout()
        plt.savefig('H2_efficiency_chart.png')
        print("Saved H2_efficiency_chart.png")
        plt.close()
    else:
        print("Could not generate H2 chart: No successful trials found.")
    
    # H3: Adaptive Behavior
    fere_df_actions = df[(df['agent_type'] == 'FERE_CRS') & (df['step_number'].notna())].copy()
    fere_df_actions['step_number'] = pd.to_numeric(fere_df_actions['step_number'])
    fere_df_actions.dropna(subset=['P_score', 'I_score'], inplace=True)

    if not fere_df_actions.empty:
        behavior_trend = fere_df_actions.groupby('step_number')[['P_score', 'I_score']].mean().reset_index()
        print("\n--- H3: Adaptive Behavior (Mean Scores per Step) ---")
        print(behavior_trend.head())
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=behavior_trend, x='step_number', y='I_score', label='Mean Informational Value (I)', marker='o', color='#C44E52')
        sns.lineplot(data=behavior_trend, x='step_number', y='P_score', label='Mean Pragmatic Value (P)', marker='o', color='#4C72B0')
        plt.title('H3: Behavioral Shift from Exploration to Exploitation')
        plt.xlabel('Step Number in Diagnosis')
        plt.ylabel('Mean Score of Chosen Action')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('H3_behavior_plot.png')
        print("Saved H3_behavior_plot.png")
        plt.close()
    else:
        print("Could not generate H3 chart: No valid FERE-CRS action data found.")


if __name__ == "__main__":
    setup_files()
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\nERROR: Please set your GOOGLE_API_KEY environment variable before running.")
    else:
        # This is now configured to run the full 500-trial experiment for the paper.
        print("--- Running full 500-trial experiment ---")
        run_experiment(num_trials_per_scenario=100)
        analyze_results()
        print("\n--- Full experiment complete. Final results and charts have been generated. ---")