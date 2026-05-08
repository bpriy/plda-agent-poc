import os
import sys
import json
import re
import requests
import time
from groq import Groq

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
        f"ACTUAL SIMULATION RESULTS:\n{results_table}\n\n"
        "TASK:\n"
        "Write a 3-paragraph, mathematically rigorous summary for a Pull Request body.\n"
        "1. BIAS & ATTENUATION: Compare point estimates. Did the Naive model exhibit attenuation bias (shrinkage toward zero)? Did the Adjusted model correct the direction of this bias?\n"
        "2. UNCERTAINTY & COVERAGE: Do NOT just look at point estimates. You must consider the Standard Errors. Evaluate if the Oracle estimate falls within the roughly 95% Confidence Interval (Estimate +/- 1.96 * SE) of the Adjusted model. Explicitly state that EM mixture adjustments theoretically inflate standard errors due to the propagation of latent linkage uncertainty.\n"
        "3. THEORETICAL CONCLUSION: Connect the massive variance inflation directly to the user's DGP (e.g., highly overlapping Beta distributions for paradata). Suggest rigorous next steps (e.g., increasing N, using stronger paradata).\n"
        "Output ONLY the text for the PR body."
    )
    
    return generate_with_retry(client, prompt)
    
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

    if "--analyze" in sys.argv:
        if not os.path.exists("results.md"):
            sys.exit(1)
            
        with open("results.md", "r") as f:
            results_table = f.read()
            
        analysis = analyze_results(client, issue_body, "results.md")
        
        with open("pr_body.md", "w") as f:
            f.write(f"## 📊 Empirical Analysis\n\n{analysis}\n\n### 📈 Simulation Results\n\n{results_table}\n\n**Linked Issue:** Closes #{issue_number}")
            
        sys.exit(0)
        
    system_knowledge = "\n### R PACKAGE DOCUMENTATION:\n"
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
                                    print(f"Direct Hit: Loaded docs for {func} from {filename}")
                    except:
                        continue
        
        if not found_files:
            print("No specific docs found. Loading package summary as fallback.")
            for filename in os.listdir(man_path):
                if "package" in filename.lower():
                    with open(os.path.join(man_path, filename), "r", encoding="utf-8") as f:
                        system_knowledge += f.read()
                        break
    else:
        system_knowledge = ""
        
    url_pattern = r'(https://github\.com/[^\s)]+/(?:files|assets)/[^\s)]+)'
    file_urls = re.findall(url_pattern, str(issue_body))
    
    attached_content = ""
    if file_urls:
        attached_content += "\n\n### SUPPLEMENTARY FILES PROVIDED BY HUMAN AUTHOR:\n"
        for url in file_urls:
            filename = url.split("/")[-1]
            print(f"Downloading supplementary file: {filename}...")
            file_text = download_and_extract_text(url)
            if file_text:
                attached_content += f"\n--- Content of {filename} ---\n{file_text}\n--- End of {filename} ---\n"

    if 'comment' in event_data and action == 'created':
        comment_body = event_data['comment']['body'].strip()
        if event_data['comment']['user']['login'] == 'github-actions[bot]': 
            sys.exit(0)

        if '/approve' in comment_body.lower():
            r_code = get_last_bot_code(repo, issue_number, github_token)
            
            # THE FIX: Ensure we found valid R code and not our error placeholder
            if r_code and not r_code.startswith("# Error"):
                os.makedirs("benchmarks", exist_ok=True)
                file_name = f"benchmarks/simulation_issue_{issue_number}.R"
                with open(file_name, "w") as f: 
                    f.write(r_code)
                sys.exit(0)
            
            # If the code parsing failed previously, block approval and warn the user
            error_msg = "🚨 **Agent Error:** I couldn't parse valid R code from my last message. Please reply asking me to revise and output the ENTIRE script in a standard ```R block."
            post_github_comment(repo, issue_number, github_token, error_msg)
            sys.exit(1)
        
        else:
            r_code = get_last_bot_code(repo, issue_number, github_token)
            if not r_code:
                error_msg = "🚨 **Agent Error**: I couldn't find any previous R code to revise. Please make sure I've already posted a methodology review."
                post_github_comment(repo, issue_number, github_token, error_msg)
                sys.exit(1)

            prompt_text = (
                "You are an expert R developer. Here is the benchmark script you previously wrote:\n"
                f"```R\n{r_code}\n```\n\n"
                f"The human researcher ran the code and provided this feedback/error:\n{comment_body}\n\n"
                "Revise the R script to fix this specific issue. \n"
                "CRITICAL CONSTRAINTS:\n"
                "1. You MUST output the ENTIRE, complete R script from start to finish. Do NOT output partial snippets or use placeholders like '# ... rest of code ...'.\n"
                "2. CAUSAL DGP & CORRUPTION: First, simulate covariates and true outcome. Second, generate a 'true_match_status' vector. Third, generate paradata (jw_score) based ONLY on 'true_match_status'. Fourth, execute a Random Swap on the outcome variable ONLY for mismatched rows to induce attenuation bias.\n"
                "3. NO CHEATING: Never pass the true 'm.rate' to adjMixture().\n"
                "4. SYNTAX: plglm() must be called with 'data = linked_data' and 'adjustment = adj_object'.\n"
                "5. TABLE OUTPUT: You MUST extract both Point Estimates and Standard Errors for all covariates (Intercept, BMI, Age). Save the Markdown table using writeLines(..., 'results.md') but NEVER put a 'collapse' argument inside writeLines.\n"
                "6. Return ONLY valid R code inside ```R blocks."
            )
            final_prompt = system_knowledge + prompt_text + attached_content
            
            response_text = generate_with_retry(client, final_prompt)
            code_match = re.search(r"```[Rr]\n(.*?)```", response_text, re.DOTALL)
            extracted_code = code_match.group(1).strip() if code_match else "# Error: Could not parse AI response."
            
            reply_body = (
                "🤖 **Agent Checkpoint: Revision**\n\n"
                "I have updated the code based on your feedback. Please review:\n\n"
                f"```R\n{extracted_code}\n```\n\n"
                "🛑 **Action Required:**\n"
                "Reply `/approve` to run this benchmark."
            )
            post_github_comment(repo, issue_number, github_token, reply_body)
            sys.exit(99) 

    elif action == 'opened':
        prompt_text = (
            "You are a rigorous statistician and R package developer for 'postlink'.\n"
            f"A human researcher submitted this benchmark request:\n{issue_body}\n\n"
            "CRITICAL STATISTICAL RULES YOU MUST FOLLOW:\n"
            "1. CAUSAL DGP & CORRUPTION: First, simulate covariates and the true outcome. Second, generate a 'true_match_status' vector (1=True, 0=False). Third, generate paradata (jw_score) based ONLY on 'true_match_status', NOT the outcome. Fourth, copy 'oracle_data' to 'linked_data' and physically execute a Random Swap on the outcome variable ONLY for rows where true_match_status == 0.\n"
            "2. MODEL SEPARATION: Fit the oracle model on 'oracle_data'. Fit the naive and adjusted models on 'linked_data'.\n"
            "3. NO CHEATING: Never pass the true 'm.rate' to adjMixture(). Let the EM algorithm estimate it.\n"
            "4. SYNTAX: plglm() must be called with 'data = linked_data' and 'adjustment = adj_object'.\n"
            "5. TABLE OUTPUT: You MUST extract both Point Estimates and Standard Errors for all covariates. Save the Markdown table using writeLines(..., 'results.md') but NEVER put a 'collapse' argument inside writeLines.\n\n"
            "Ensure you follow any methodology outlined in the supplementary files provided below.\n"
            "Return ONLY valid R code inside ```R blocks."
        )
        final_prompt = system_knowledge + prompt_text + attached_content
        
        response_text = generate_with_retry(client, final_prompt)
        code_match = re.search(r"```[Rr]\n(.*?)```", response_text, re.DOTALL)
        extracted_code = code_match.group(1).strip() if code_match else "# Error: Could not parse AI response."
            
        reply_body = (
            "🤖 **Agent Checkpoint: Methodology Review**\n\n"
            "I have designed the following Data Generating Process (DGP) and benchmark suite "
            "based on your specifications and authored files. Please review:\n\n"
            f"```R\n{extracted_code}\n```\n\n"
            "🛑 **Action Required:**\n"
            "Reply `/approve` to execute this benchmark, or reply with changes."
        )
        post_github_comment(repo, issue_number, github_token, reply_body)
        sys.exit(99)

if __name__ == "__main__":
    main()
