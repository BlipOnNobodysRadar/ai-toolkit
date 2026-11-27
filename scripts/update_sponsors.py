import os
from urllib.parse import urlsplit
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

# API credentials
PATREON_TOKEN = os.getenv("PATREON_ACCESS_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_ORG = os.getenv("GITHUB_ORG")  # Organization name (optional)

# Output file
README_PATH = "SUPPORTERS.md"

def fetch_patreon_supporter_count():
    """Count active patrons without collecting names or profile pictures."""
    print("Fetching Patreon supporters...")
    
    headers = {
        "Authorization": f"Bearer {PATREON_TOKEN}",
        "Content-Type": "application/json"
    }
    
    url = "https://www.patreon.com/api/oauth2/v2/campaigns"
    
    try:
        # First get the campaign ID
        campaign_response = requests.get(url, headers=headers)
        campaign_response.raise_for_status()
        campaign_data = campaign_response.json()
        
        if not campaign_data.get('data'):
            print("No campaigns found for this Patreon account")
            return 0
        
        campaign_id = campaign_data['data'][0]['id']
        
        members_url = f"https://www.patreon.com/api/oauth2/v2/campaigns/{campaign_id}/members"
        params = {"fields[member]": "patron_status"}
        count = 0
        while members_url:
            target = urlsplit(members_url)
            if target.scheme != "https" or target.netloc.lower() not in {"www.patreon.com", "www.patreon.com:443"}:
                raise requests.exceptions.RequestException("Unexpected Patreon pagination origin")
            members_response = requests.get(members_url, headers=headers, params=params)
            members_response.raise_for_status()
            members_data = members_response.json()
            
            count += sum(
                member.get("attributes", {}).get("patron_status") == "active_patron"
                for member in members_data.get("data", [])
            )
            members_url = members_data.get("links", {}).get("next")
        print(f"Found {count} active Patreon supporters")
        return count
    except requests.exceptions.RequestException as e:
        detail = e.response.status_code if e.response is not None else type(e).__name__
        print(f"Error fetching Patreon supporter count ({detail})")
        return 0

def fetch_github_sponsors():
    """Fetch current GitHub sponsors for a user or organization"""
    print("Fetching GitHub sponsors...")
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Determine if we're fetching for a user or an organization
    entity_type = "organization" if GITHUB_ORG else "user"
    entity_name = GITHUB_ORG if GITHUB_ORG else GITHUB_USERNAME
    
    if not entity_name:
        print("Error: Neither GITHUB_USERNAME nor GITHUB_ORG is set")
        return []
    
    # Different GraphQL query structure based on entity type
    if entity_type == "user":
        query = """
        query {
          user(login: "%s") {
            sponsorshipsAsMaintainer(first: 100) {
              nodes {
                sponsorEntity {
                  ... on User {
                    login
                    name
                    avatarUrl
                    url
                  }
                  ... on Organization {
                    login
                    name
                    avatarUrl
                    url
                  }
                }
                tier {
                  monthlyPriceInDollars
                }
                privacyLevel
                isActive
              }
            }
          }
        }
        """ % entity_name
    else:  # organization
        query = """
        query {
          organization(login: "%s") {
            sponsorshipsAsMaintainer(first: 100) {
              nodes {
                sponsorEntity {
                  ... on User {
                    login
                    name
                    avatarUrl
                    url
                  }
                  ... on Organization {
                    login
                    name
                    avatarUrl
                    url
                  }
                }
                tier {
                  monthlyPriceInDollars
                }
                privacyLevel
                isActive
              }
            }
          }
        }
        """ % entity_name
    
    try:
        response = requests.post(
            "https://api.github.com/graphql",
            headers=headers,
            json={"query": query}
        )
        response.raise_for_status()
        data = response.json()
        
        # Process the response - the path to the data differs based on entity type
        if entity_type == "user":
            sponsors_data = data.get('data', {}).get('user', {}).get('sponsorshipsAsMaintainer', {}).get('nodes', [])
        else:
            sponsors_data = data.get('data', {}).get('organization', {}).get('sponsorshipsAsMaintainer', {}).get('nodes', [])
        
        sponsors = []
        for sponsor in sponsors_data:
            # This list is published. The maintainer's token can also see private sponsors.
            if sponsor.get('isActive') and sponsor.get('privacyLevel') == 'PUBLIC':
                entity = sponsor.get('sponsorEntity', {})
                name = entity.get('name') or entity.get('login', 'Anonymous Sponsor')
                profile_image = entity.get('avatarUrl')
                profile_url = entity.get('url')
                amount = sponsor.get('tier', {}).get('monthlyPriceInDollars', 0)
                
                sponsors.append({
                    'name': name,
                    'profile_image': profile_image,
                    'profile_url': profile_url,
                    'platform': 'GitHub Sponsors',
                    'amount': amount
                })
        
        print(f"Found {len(sponsors)} active GitHub sponsors for {entity_type} '{entity_name}'")
        return sponsors
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching GitHub sponsors data: {e}")
        return []

def generate_readme(supporters, patreon_count=0):
    """Generate a README.md file with supporter information"""
    print(f"Generating {README_PATH}...")
    
    # Sort supporters by amount (descending) and then by name
    supporters.sort(key=lambda x: (-x['amount'], x['name'].lower()))
    
    # Determine the proper footer links based on what's configured
    github_entity = GITHUB_ORG if GITHUB_ORG else GITHUB_USERNAME
    github_entity_type = "orgs" if GITHUB_ORG else "sponsors"
    github_sponsor_url = f"https://github.com/{github_entity_type}/{github_entity}"
    
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write("## Support My Work\n\n")
        f.write("If you enjoy my work, or use it for commercial purposes, please consider sponsoring me so I can continue to maintain it. Every bit helps! \n\n")
        # Create appropriate call-to-action based on what's configured
        cta_parts = []
        if github_entity:
            cta_parts.append(f"[Become a sponsor on GitHub]({github_sponsor_url})")
        if PATREON_TOKEN:
            cta_parts.append("[support me on Patreon](https://www.patreon.com/ostris)")
        
        if cta_parts:
            if GITHUB_ORG:
                f.write(f"{' or '.join(cta_parts)}.\n\n")
        f.write("Thank you to all my current supporters!\n\n")
        
        f.write(f"_Last updated: {datetime.now().strftime('%Y-%m-%d')}_\n\n")
        
        # Write GitHub Sponsors section
        github_sponsors = [s for s in supporters if s['platform'] == 'GitHub Sponsors']
        if github_sponsors:
            f.write("### GitHub Sponsors\n\n")
            for sponsor in github_sponsors:
                if sponsor['profile_image']:
                    f.write(f"<a href=\"{sponsor['profile_url']}\" title=\"{sponsor['name']}\"><img src=\"{sponsor['profile_image']}\" width=\"50\" height=\"50\" alt=\"{sponsor['name']}\" style=\"border-radius:50%;display:inline-block;\"></a> ")
                else:
                    f.write(f"[{sponsor['name']}]({sponsor['profile_url']}) ")
            f.write("\n\n")
        
        # Membership data carries no consent for publishing a patron's name.
        if patreon_count:
            f.write("### Patreon Supporters\n\n")
            f.write(f"Thank you to our {patreon_count} active Patreon supporters.\n\n")

        f.write("\n---\n\n")
        
    
    print(f"Successfully generated {README_PATH} with {len(supporters) + patreon_count} supporters!")

def main():
    """Main function"""
    print("Starting supporter data collection...")
    
    # Check if required environment variables are set
    missing_vars = []
    if not GITHUB_TOKEN:
        missing_vars.append("GITHUB_TOKEN")
    
    # Either username or org is required for GitHub
    if not GITHUB_USERNAME and not GITHUB_ORG:
        missing_vars.append("GITHUB_USERNAME or GITHUB_ORG")
    
    # Patreon token is optional but warn if missing
    patreon_enabled = bool(PATREON_TOKEN)
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please add them to your .env file")
        return
    
    if not patreon_enabled:
        print("Warning: PATREON_ACCESS_TOKEN not set. Will only fetch GitHub sponsors.")
    
    # Fetch data from both platforms
    patreon_count = fetch_patreon_supporter_count() if PATREON_TOKEN else 0
    github_sponsors = fetch_github_sponsors()
    
    if not patreon_count and not github_sponsors:
        print("No supporters found on either platform")
        return
    
    # Generate README
    generate_readme(github_sponsors, patreon_count)

if __name__ == "__main__":
    main()