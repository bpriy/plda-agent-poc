import os
import sys
import json
import re
import requests
import time
from groq import Groq

# ==========================================
# CORE API WRAPPERS
# ==========================================
def generate_with_retry(client, prompt_text, max_retries=3):
    """A fault-tolerant wrapper that calls Groq's Llama 3 model."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "503" in str(e):
                wait_time = (attempt + 1) * 5
                print(f"API busy. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            print(f"API Request Failed: {e}")
            raise e

def download_and_extract_text(url):
    """Downloads an attached file from GitHub and extracts its text."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content.decode('utf-8')
    except Exception as e:
        print(f"Failed to extract text from {url}: {e}")
        return None

def post_github_comment(repo, issue_number, token, body):
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}", 
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.post(url, headers=headers, json={"body": body})
    response.raise_for_status()

def get_last_bot_code(repo, issue_number, token):
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}", 
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers).json()
    bot_comments = [c for c in response if c['user']['login'] == 'github-actions[bot]']
    if not bot_comments: return None
    last_comment = bot_comments[-1]['body']
    code_match = re.search(r"```[Rr]\n(.*?)```", last_comment, re.DOTALL)
    if code_match: return code_match.group(1).strip()
    return None

def analyze_results(client, issue_body, results_path):
    """Reads the actual simulation results and writes an objective critique."""
    with open(results_path, "r") as f:
        results_table = f.read()

    prompt = (
        "You are an objective, rigorous academic statistician, peer reviewer, and expert in record linkage, post-linkage data analysis, and the postlink R package. Analyze these simulation results "
        "for a Record Linkage adjustment model (postlink package).\n\n"
        f"USER'S ORIGINAL INTENT:\n{issue_body}\n\n"
        f"ACTUAL SIMULATION RESULTS (Sensitivity Sweep):\n{results_table}\n\n"
        "TASK:\n"
        "Write a 3-paragraph, mathematically rigorous summary for a Pull Request body.\n"
        "1. BIAS & ATTENUATION: Compare point estimates across the different mismatch rates. Did the Naive model exhibit worsening attenuation bias as error rates increased? Did the Adjusted model correct it?\n"
        "2. UNCERTAINTY & COVERAGE: Evaluate standard errors. State explicitly that EM mixture adjustments theoretically inflate standard errors due to the propagation of latent linkage uncertainty.\n"
        "3. THEORETICAL CONCLUSION: Connect the variance inflation directly to the user's DGP. Evaluate the package's boundary performance at the highest mismatch rate.\n"
        "Output ONLY the text for the PR body."
    )
    return generate_with_retry(client, prompt)

# ==========================================
# THE AGENTIC COUNCIL (Highest Level Logic)
# ==========================================
def conduct_council(client, issue_body, system_knowledge):
    """Orchestrates the three-agent deliberation loop."""
    
    # --- PHASE 1: THE ARCHITECT ---
    architect_prompt = (
        "You are the Lead PhD Statistician. Design a rigorous simulation plan for this request:\n"
        f"ISSUE: {issue_body}\n\n"
        "REQUIREMENTS:\n"
        "1. Identify the outcome type and select the appropriate GLM family (e.g., gaussian, binomial, poisson).\n"
        "2. Define the Causal DAG (Covariates -> True Outcome -> Match Status -> Paradata).\n"
        "3. Plan a Sensitivity Sweep: We must test mismatch rates of 0.1, 0.3, and 0.5 in a loop.\n"
        "4. Define the outcome corruption (Random Swap within the mismatched subset).\n"
        "Return ONLY the statistical design plan."
    )
    plan = generate_with_retry(client, architect_prompt + system_knowledge)

    # --- PHASE 2: THE DEVELOPER ---
    developer_prompt = (
        "You are a Senior R Developer. Translate this statistical plan into a formal 'testthat' suite:\n"
        f"PLAN: {plan}\n\n"
        "CONSTRAINTS:\n"
        "1. Wrap the logic in: test_that('Simulation Sweep', { ... })\n"
        "2. Include 'set.seed(123)' and 'skip_on_cran()' inside the test block.\n"
        "3. Loop through the sensitivity sweep (mismatch = 0.1, 0.3, 0.5).\n"
        "4. NO CHEATING: Do not pass the true mismatch rate to adjMixture(). Let the EM estimate it.\n"
        "5. EXTRACT: Use broom::tidy() to easily extract point estimates and standard errors for all models across all loops.\n"
        "6. OUTPUT: Save a single cohesive Markdown table comparing the models at different mismatch rates to 'results.md' using writeLines(). Do not put a 'collapse' argument inside writeLines.\n"
        "Return ONLY valid R code in a ```R block."
    )
    code_response = generate_with_retry(client, developer_prompt + system_knowledge)
    code_match = re.search(r"```[Rr]\n(.*?)```", code_response, re.DOTALL)
    code = code_match.group(1).strip() if code_match else code_response

    # --- PHASE 3: THE CRITIC (Adversarial Loop) ---
    critique = "PASSED"
    for attempt in range(2):
        critic_prompt = (
            "You are a Skeptical Peer Reviewer evaluating an R simulation script.\n"
            f"CODE:\n```R\n{code}\n```\n\n"
            "CHECK FOR FATAL FLAWS:\n"
            "1. DATA LEAKAGE: Is the true mismatch rate passed to the `m.rate` argument of adjMixture? (This is cheating).\n"
            "2. CAUSALITY FAULT: Is 'jw_score' generated based on Disease_Status instead of Match_Status?\n"
            "3. SYNTAX: Is `writeLines` used correctly without a `collapse` argument inside the function call itself?\n"
            "4. COMPLETENESS: Does the script extract Standard Errors?\n"
            "If the code is flawless, reply exactly with 'PASSED'. Otherwise, list the specific errors."
        )
        critique = generate_with_retry(client, critic_prompt)
        
        if "PASSED" in critique:
            break
        else:
            # Re-run Developer to fix the code
            fix_prompt = (
                "You are a Senior R Developer. Fix your script based on this code review:\n"
                f"ERRORS:\n{critique}\n\n"
                f"ORIGINAL CODE:\n```R\n{code}\n```\n\n"
                "Return ONLY the fixed R code in a ```R block."
            )
            code_response = generate_with_retry(client, fix_prompt)
            code_match = re.search(r"```[Rr]\n(.*?)```", code_response, re.DOTALL)
            code = code_match.group(1).strip() if code_match else code_response

    return plan, critique, code

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def main():
    groq_key = os.environ.get("GROQ_API_KEY")
    github_token = os.environ.get("GITHUB_TOKEN")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    
    if not all([groq_key, github_token, event_path]):
        print("Missing required environment variables.")
        sys.exit(1)
        
    client = Groq(api_key=groq_key)
    with open(event_path, 'r') as f: 
        event_data = json.load(f)

    repo = event_data['repository']['full_name']
    issue_number = event_data['issue']['number']
    issue_body = event_data['issue']['body']
    action = event_data.get('action')

    # ---------------------------------------------------------
    # JOB 3: PR ANALYSIS MODE
    # ---------------------------------------------------------
    if "--analyze" in sys.argv:
        if not os.path.exists("results.md"):
            sys.exit(1)
            
        with open("results.md", "r") as f:
            results_table = f.read()
            
        analysis = analyze_results(client, issue_body, "results.md")
        
        with open("pr_body.md", "w") as f:
            f.write(f"## 📊 Empirical Analysis\n\n{analysis}\n\n### 📈 Sensitivity Sweep Results\n\n{results_table}\n\n**Linked Issue:** Closes #{issue_number}")
            
        sys.exit(0)
        
    # ---------------------------------------------------------
    # KNOWLEDGE RETRIEVAL
    # ---------------------------------------------------------
    system_knowledge = "\n\n### SYSTEM CONTEXT (R Package Docs):\n"
    man_path = "man"
    
    potential_functions = re.findall(r'\b([a-zA-Z0-9_.]+)\s*\(', issue_body)
    potential_functions.append("adjMixture") 
    functions_to_lookup = list(set(potential_functions))
    found_files = set() 
    
    if os.path.exists(man_path) and os.path.isdir(man_path):
        for func in functions_to_lookup:
            for filename in os.listdir(man_path):
                file_path = os.path.join(man_path, filename)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if func.lower() in filename.lower() or f"\\alias{{{func}}}" in content:
                                if filename not in found_files:
                                    system_knowledge += f"\n--- DOCS FROM {filename} ---\n{content}\n"
                                    found_files.add(filename)
                    except:
                        continue
        
        if not found_files:
            for filename in os.listdir(man_path):
                if "package" in filename.lower():
                    with open(os.path.join(man_path, filename), "r", encoding="utf-8") as f:
                        system_knowledge += f.read()
                        break
    
    url_pattern = r'(https://github\.com/[^\s)]+/(?:files|assets)/[^\s)]+)'
    file_urls = re.findall(url_pattern, str(issue_body))
    if file_urls:
        system_knowledge += "\n\n### SUPPLEMENTARY FILES PROVIDED BY HUMAN:\n"
        for url in file_urls:
            file_text = download_and_extract_text(url)
            if file_text:
                system_knowledge += f"\n--- Content of {url.split('/')[-1]} ---\n{file_text}\n"

    # ---------------------------------------------------------
    # STATE 3: EXECUTION MODE (/approve) & REVISION
    # ---------------------------------------------------------
    if 'comment' in event_data and action == 'created':
        comment_body = event_data['comment']['body'].strip()
        if event_data['comment']['user']['login'] == 'github-actions[bot]': 
            sys.exit(0)

        # TRIGGER BENCHMARK RUN
        if '/approve' in comment_body.lower():
            r_code = get_last_bot_code(repo, issue_number, github_token)
            
            if r_code and not r_code.startswith("# Error"):
                os.makedirs("tests/testthat", exist_ok=True)
                file_name = f"tests/testthat/test-issue-{issue_number}.R"
                with open(file_name, "w") as f: 
                    f.write(r_code)
                sys.exit(0)
            
            error_msg = "🚨 **Agent Error:** I couldn't parse valid R code from my last message. Please ask me to revise."
            post_github_comment(repo, issue_number, github_token, error_msg)
            sys.exit(1)
        
        # TRIGGER REVISION
        else:
            r_code = get_last_bot_code(repo, issue_number, github_token)
            if not r_code:
                post_github_comment(repo, issue_number, github_token, "🚨 I couldn't find previous R code to revise.")
                sys.exit(1)

            # Fast-track revision directly to Developer
            prompt_text = (
                "You are an expert R developer. The user rejected your last testthat script:\n"
                f"```R\n{r_code}\n```\n\n"
                f"USER FEEDBACK: {comment_body}\n\n"
                "Fix the script. Ensure you output the ENTIRE script inside a ```R block."
            )
            response_text = generate_with_retry(client, prompt_text + system_knowledge)
            code_match = re.search(r"```[Rr]\n(.*?)```", response_text, re.DOTALL)
            extracted_code = code_match.group(1).strip() if code_match else "# Error: Could not parse AI response."
            
            reply_body = (
                "🤖 **Agent Checkpoint: Revision**\n\n"
                "I have updated the code based on your feedback:\n\n"
                f"```R\n{extracted_code}\n```\n\n"
                "🛑 **Action Required:** Reply `/approve` to run this benchmark."
            )
            post_github_comment(repo, issue_number, github_token, reply_body)
            sys.exit(99) 

    # ---------------------------------------------------------
    # STATE 1: DRAFTING MODE (New Issue Opened)
    # ---------------------------------------------------------
    elif action == 'opened':
        plan, critique, code = conduct_council(client, issue_body, system_knowledge)
        
        reply_body = (
            "## 🏛️ Agentic Council Deliberation\n\n"
            f"### 🧠 Statistical Design (The Architect)\n{plan}\n\n"
            f"### 🔍 Peer Review (The Critic)\n{critique}\n\n"
            "### 💻 Proposed Implementation (The Developer)\n"
            f"```R\n{code}\n```\n\n"
            "---\n"
            "🛑 **Action Required:** Reply `/approve` to merge this test suite into `tests/testthat/` and execute the simulation."
        )
        post_github_comment(repo, issue_number, github_token, reply_body)
        sys.exit(99)

if __name__ == "__main__":
    main()
