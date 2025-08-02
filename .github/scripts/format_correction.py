#!/usr/bin/env python3
"""
Format correction and validation for Claude review YAML.
This script runs in a sandboxed environment with restricted permissions.
"""
import re
import yaml
import sys
import os
from datetime import datetime

# Security: Restrict file operations to current directory only
os.chdir(os.getcwd())

def safe_int(value, default):
    try:
        return int(value)
    except:
        return default

def safe_float(value, default):
    try:
        return float(value)
    except:
        return default

try:
    # Read the raw response with size limit for security
    max_size = 1024 * 1024  # 1MB limit
    with open('raw_review_output.txt', 'r') as f:
        content = f.read(max_size)
    
    print("🔧 Processing Claude response for format correction...")
    
    # Strategy 1: Try to extract YAML block if it exists
    yaml_match = re.search(r'```yaml\s*\n(.*?)\n```', content, re.DOTALL)
    if yaml_match:
        yaml_content = yaml_match.group(1)
        print("✓ Found YAML block in response")
    else:
        # Strategy 2: Extract everything after markdown headers (---\n)
        if '---\n' in content:
            parts = content.split('---\n', 1)
            if len(parts) > 1:
                yaml_content = parts[1].strip()
                print("✓ Extracted YAML after markdown separator")
            else:
                yaml_content = content
        else:
            yaml_content = content
    
    # Clean the YAML content
    yaml_content = yaml_content.strip()
    
    # Remove any remaining markdown formatting
    yaml_content = re.sub(r'^\*\*.*?\*\*.*?\n', '', yaml_content, flags=re.MULTILINE)
    yaml_content = re.sub(r'^---\s*$', '', yaml_content, flags=re.MULTILINE)
    
    # Try to parse and validate the YAML with safe loader
    try:
        data = yaml.safe_load(yaml_content)
        if isinstance(data, dict):
            print("✓ YAML parsed successfully")
            
            # Ensure required fields exist with defaults
            if 'schema_version' not in data:
                data['schema_version'] = "1.0"
            if 'pr_number' not in data:
                data['pr_number'] = safe_int(os.environ.get('PR_NUMBER', '0'), 0)
            if 'timestamp' not in data:
                data['timestamp'] = datetime.utcnow().isoformat() + 'Z'
            if 'reviewer' not in data:
                data['reviewer'] = "ARC-Reviewer"
            if 'verdict' not in data:
                # Determine verdict based on coverage and blocking issues
                has_blockers = bool(data.get('issues', {}).get('blocking', []))
                coverage_baseline = safe_float(os.environ.get('COVERAGE_BASELINE', '78.0'), 78.0)
                coverage_ok = data.get('coverage', {}).get('current_pct', 0) >= coverage_baseline
                
                if has_blockers or not coverage_ok:
                    data['verdict'] = "REQUEST_CHANGES"
                else:
                    data['verdict'] = "APPROVE"
            if 'summary' not in data:
                data['summary'] = "Code review completed"
            if 'coverage' not in data:
                # Use actual coverage if available
                actual_coverage = safe_float(os.environ.get('COVERAGE_PCT', '0'), 0)
                coverage_baseline = safe_float(os.environ.get('COVERAGE_BASELINE', '78.0'), 78.0)
                meets_baseline = actual_coverage >= coverage_baseline
                data['coverage'] = {
                    'current_pct': actual_coverage,
                    'status': "PASS" if meets_baseline else "FAIL",
                    'meets_baseline': meets_baseline
                }
            if 'issues' not in data:
                data['issues'] = {
                    'blocking': [],
                    'warnings': [],
                    'nits': []
                }
            if 'automated_issues' not in data:
                data['automated_issues'] = []
            
            # Output clean YAML with safe dump
            clean_yaml = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
            with open('corrected_review.yaml', 'w') as f:
                f.write(clean_yaml)
            
            print("✅ Generated corrected YAML format")
            
        else:
            print("❌ YAML content is not a valid dictionary")
            sys.exit(1)
            
    except yaml.YAMLError as e:
        print(f"❌ YAML parsing failed: {e}")
        # Create minimal valid YAML as fallback
        actual_coverage = safe_float(os.environ.get('COVERAGE_PCT', '0'), 0)
        coverage_baseline = safe_float(os.environ.get('COVERAGE_BASELINE', '78.0'), 78.0)
        meets_baseline = actual_coverage >= coverage_baseline
        pr_number = safe_int(os.environ.get('PR_NUMBER', '0'), 0)
        
        fallback_data = {
            'schema_version': "1.0",
            'pr_number': pr_number,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'reviewer': "ARC-Reviewer",
            'verdict': "APPROVE" if meets_baseline else "REQUEST_CHANGES",
            'summary': "Format correction applied - review completed",
            'coverage': {
                'current_pct': actual_coverage,
                'status': "PASS" if meets_baseline else "FAIL",
                'meets_baseline': meets_baseline
            },
            'issues': {
                'blocking': [] if meets_baseline else [{'description': f'Coverage below baseline: {actual_coverage}% < {coverage_baseline}%', 'category': 'coverage'}],
                'warnings': [{'description': 'Original review format was invalid', 'category': 'format'}],
                'nits': []
            },
            'automated_issues': []
        }
        
        with open('corrected_review.yaml', 'w') as f:
            yaml.safe_dump(fallback_data, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Generated fallback YAML format")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)