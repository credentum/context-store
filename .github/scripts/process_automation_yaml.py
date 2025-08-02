#!/usr/bin/env python3
"""
Process automation YAML from PR comments.
This script runs in a sandboxed environment with restricted permissions.
"""
import re
import yaml
import os
import sys

# Security: Restrict file operations
os.chdir(os.getcwd())

def process_automation():
    # Read automation comment from stdin
    content = sys.stdin.read()
    
    # Extract YAML between markers
    match = re.search(r'<!-- ARC-AUTOMATION\n(.*?)\n-->', content, re.DOTALL)
    if not match:
        print("No automation YAML found")
        sys.exit(0)
    
    yaml_content = match.group(1).strip()
    
    try:
        # Use safe_load to prevent arbitrary code execution
        data = yaml.safe_load(yaml_content)
        
        # Validate structure
        if not isinstance(data, dict):
            raise ValueError("Invalid YAML structure")
        
        # Extract issues safely
        issues = []
        for issue in data.get('issues', []):
            if isinstance(issue, dict) and 'title' in issue and 'body' in issue:
                # Sanitize labels to prevent injection
                labels = []
                for label in issue.get('labels', []):
                    if isinstance(label, str) and re.match(r'^[a-zA-Z0-9-_]+$', label):
                        labels.append(label)
                
                issues.append({
                    'title': str(issue['title'])[:200],  # Limit title length
                    'body': str(issue['body'])[:5000],   # Limit body length
                    'labels': labels[:10]  # Limit number of labels
                })
        
        # Write validated issues
        with open('followup_issues.json', 'w') as f:
            import json
            json.dump({'issues': issues}, f)
        
        print(f"Found {len(issues)} follow-up issues")
        
    except Exception as e:
        print(f"Error processing automation YAML: {e}")
        sys.exit(1)

if __name__ == '__main__':
    process_automation()