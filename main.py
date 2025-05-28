from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import whois
import datetime
import hashlib
import shodan
import dns.resolver
import ssl
import socket
from OpenSSL import SSL
import os
import time
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# API Keys
VT_API_KEY = '99e4922915b2a1c753dfd66e541d41df6a3522cb906b6c0d6ae7c1df6f529ae5'  # Your VirusTotal API key
SHODAN_API_KEY = '3pEzEzIv84bpdS6e3fvftB3d85RA8GZb'  # Your Shodan API key


@app.route('/')
def index():
    return render_template('inputs.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/analyze', methods=['POST'])
def analyze_threats():
    results = {}

    # Phishing Detection
    if 'phishing-url' in request.form and request.form['phishing-url']:
        url = request.form['phishing-url']
        results['phishing_url'] = scan_url_internal(url)

    if 'email-content' in request.form and request.form['email-content']:
        email_content = request.form['email-content']
        results['email_analysis'] = analyze_email_content(email_content)

    if 'domain-age' in request.form and request.form['domain-age']:
        domain = request.form['domain-age']
        results['domain_age'] = check_domain_age(domain)

    # Malware Analysis
    if 'file-upload' in request.files and request.files['file-upload'].filename != '':
        uploaded_file = request.files['file-upload']
        results['file_scan'] = scan_uploaded_file(uploaded_file)

    if 'file-hash' in request.form and request.form['file-hash']:
        file_hash = request.form['file-hash']
        results['hash_lookup'] = check_file_hash(file_hash)

    # Network Threat Detection
    if 'ip-check' in request.form and request.form['ip-check']:
        ip_address = request.form['ip-check']
        results['ip_reputation'] = check_ip_reputation_shodan(ip_address)

    if 'port-scan' in request.form and request.form['port-scan']:
        ip_address = request.form['port-scan']
        results['port_scan'] = scan_ports(ip_address)

    if 'dns-history' in request.form and request.form['dns-history']:
        domain = request.form['dns-history']
        results['dns_history'] = check_dns_history(domain)

    if 'ssl-check' in request.form and request.form['ssl-check']:
        url = request.form['ssl-check']
        results['ssl_check'] = check_ssl_certificate(url)

    # Credential Protection
    if 'password-check' in request.form and request.form['password-check']:
        password = request.form['password-check']
        results['password_breach'] = check_password_breach(password)

    # Advanced Threat Detection
    if 'threat-report' in request.form and request.form['threat-report']:
        indicator = request.form['threat-report']
        results['threat_intel'] = generate_threat_report(indicator)

    if 'cve-search' in request.form and request.form['cve-search']:
        software = request.form['cve-search']
        results['cve_search'] = search_cve(software)

    if 'geolocation-ip' in request.form and request.form['geolocation-ip']:
        ip_address = request.form['geolocation-ip']
        results['geolocation'] = get_ip_geolocation(ip_address)

    return jsonify(results)


# Phishing Detection Functions
def scan_url_internal(url):
    """Scan URL using VirusTotal"""
    try:
        headers = {'x-apikey': VT_API_KEY}
        scan_url = 'https://www.virustotal.com/api/v3/urls'
        response = requests.post(scan_url, headers=headers, data={'url': url})

        if response.status_code != 200:
            return {'error': 'Failed to submit URL for scanning'}

        scan_data = response.json()
        analysis_id = scan_data['data']['id']
        analysis_url = f'https://www.virustotal.com/api/v3/analyses/{analysis_id}'

        while True:
            analysis_response = requests.get(analysis_url, headers=headers)
            analysis_data = analysis_response.json()
            if analysis_data['data']['attributes']['status'] == 'completed':
                break
            time.sleep(5)

        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip('=')
        report_url = f'https://www.virustotal.com/api/v3/urls/{url_id}'
        report_response = requests.get(report_url, headers=headers)

        if report_response.status_code != 200:
            return {'error': 'Failed to get URL report'}

        report_data = report_response.json()
        attributes = report_data['data']['attributes']
        stats = attributes['last_analysis_stats']

        threats = []
        for engine, result in attributes['last_analysis_results'].items():
            if result['category'] in ['malicious', 'suspicious']:
                threats.append(f"{engine}: {result['result']}")

        return {
            'url': url,
            'malicious': stats.get('malicious', 0),
            'suspicious': stats.get('suspicious', 0),
            'harmless': stats.get('harmless', 0),
            'undetected': stats.get('undetected', 0),
            'threat_list': threats,
            'reputation': attributes.get('reputation', 0),
            'categories': attributes.get('categories', {}),
            'last_analysis_date': attributes.get('last_analysis_date', 0)
        }
    except Exception as e:
        return {'error': str(e)}


def analyze_email_content(content):
    """Analyze email content for phishing patterns"""
    phishing_keywords = ['urgent', 'password', 'verify', 'account', 'suspended', 'immediately']
    score = 0
    matches = []

    content_lower = content.lower()
    for keyword in phishing_keywords:
        if keyword in content_lower:
            score += 1
            matches.append(keyword)

    return {
        'phishing_score': score,
        'matched_keywords': matches,
        'is_suspicious': score > 2
    }


def check_domain_age(domain):
    """Check when a domain was registered"""
    try:
        domain_info = whois.whois(domain)
        creation_date = domain_info.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        age = (datetime.datetime.now() - creation_date).days if creation_date else None
        return {
            'domain': domain,
            'creation_date': str(creation_date),
            'age_days': age,
            'is_new': age is not None and age < 30
        }
    except Exception as e:
        return {'error': str(e)}


# Malware Analysis Functions
def scan_uploaded_file(file):
    """Scan uploaded file with VirusTotal"""
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Upload file to VirusTotal
        with open(filepath, 'rb') as f:
            files = {'file': (filename, f)}
            headers = {'x-apikey': VT_API_KEY}
            response = requests.post('https://www.virustotal.com/api/v3/files', files=files, headers=headers)

        if response.status_code != 200:
            return {'error': 'VirusTotal upload failed'}

        analysis_id = response.json()['data']['id']

        # Wait for analysis (polling)
        headers = {'x-apikey': VT_API_KEY}
        result_url = f'https://www.virustotal.com/api/v3/analyses/{analysis_id}'
        while True:
            result_response = requests.get(result_url, headers=headers)
            result_data = result_response.json()
            if result_data['data']['attributes']['status'] == 'completed':
                break
            time.sleep(2)

        # Get detailed file report
        file_id = result_data['meta']['file_info']['sha256']
        report_url = f'https://www.virustotal.com/api/v3/files/{file_id}'
        report_response = requests.get(report_url, headers=headers)
        report_data = report_response.json()['data']

        attr = report_data['attributes']
        stats = attr['last_analysis_stats']
        engines = report_data['attributes']['last_analysis_results']

        threats = [f"{name}: {r['result']}" for name, r in engines.items() if
                   r['category'] in ['malicious', 'suspicious']]

        return {
            'filename': filename,
            'filetype': file.mimetype,
            'size': round(os.path.getsize(filepath) / (1024 * 1024), 2),
            'md5': attr.get('md5'),
            'sha1': attr.get('sha1'),
            'sha256': attr.get('sha256'),
            'first_submission': attr.get('first_submission_date'),
            'last_submission': attr.get('last_submission_date'),
            'scan_date': attr.get('last_analysis_date'),
            'total_engines': stats.get('harmless', 0) + stats.get('malicious', 0) + stats.get('suspicious', 0),
            'malicious': stats.get('malicious', 0),
            'suspicious': stats.get('suspicious', 0),
            'harmless': stats.get('harmless', 0),
            'undetected': stats.get('undetected', 0),
            'threat_list': threats[:10]  # limit to 10 threats
        }
    except Exception as e:
        return {'error': str(e)}


def check_file_hash(file_hash):
    """Check file hash against VirusTotal"""
    try:
        headers = {'x-apikey': VT_API_KEY}
        report_url = f'https://www.virustotal.com/api/v3/files/{file_hash}'
        report_response = requests.get(report_url, headers=headers)

        if report_response.status_code != 200:
            return {'error': 'File not found in VirusTotal database'}

        report_data = report_response.json()['data']
        attr = report_data['attributes']
        stats = attr['last_analysis_stats']
        engines = attr['last_analysis_results']

        threats = [f"{name}: {r['result']}" for name, r in engines.items() if
                   r['category'] in ['malicious', 'suspicious']]

        return {
            'hash': file_hash,
            'malicious': stats.get('malicious', 0),
            'suspicious': stats.get('suspicious', 0),
            'harmless': stats.get('harmless', 0),
            'undetected': stats.get('undetected', 0),
            'threat_list': threats[:10],
            'names': attr.get('names', []),
            'type_description': attr.get('type_description', 'Unknown'),
            'last_analysis_date': attr.get('last_analysis_date', 0)
        }
    except Exception as e:
        return {'error': str(e)}


# Network Threat Detection Functions
def check_ip_reputation_shodan(ip_address):
    """Check IP reputation using Shodan"""
    try:
        api = shodan.Shodan(SHODAN_API_KEY)
        host = api.host(ip_address)

        return {
            'ip': host['ip_str'],
            'country': host.get('country_name', 'Unknown'),
            'city': host.get('city', 'Unknown'),
            'org': host.get('org', 'Unknown'),
            'isp': host.get('isp', 'Unknown'),
            'last_update': host.get('last_update', 'Unknown'),
            'ports': host.get('ports', []),
            'vulnerabilities': host.get('vulns', []),
            'tags': host.get('tags', []),
            'hostnames': host.get('hostnames', []),
            'asn': host.get('asn', 'Unknown'),
            'services': [{
                'port': item['port'],
                'transport': item['transport'],
                'service': item.get('service', 'Unknown'),
                'product': item.get('product', 'Unknown'),
                'version': item.get('version', 'Unknown')
            } for item in host.get('data', [])]
        }
    except shodan.APIError as e:
        return {'error': str(e)}
    except Exception as e:
        return {'error': str(e)}


def scan_ports(ip_address):
    """Scan for open ports using Shodan"""
    try:
        api = shodan.Shodan(SHODAN_API_KEY)
        host = api.host(ip_address)

        return {
            'ip': host['ip_str'],
            'ports': host.get('ports', []),
            'services': [{
                'port': item['port'],
                'transport': item['transport'],
                'service': item.get('service', 'Unknown'),
                'banner': item.get('data', '')
            } for item in host.get('data', [])]
        }
    except shodan.APIError as e:
        if 'No information available' in str(e):
            return {'error': 'No Shodan data available for this IP'}
        return {'error': str(e)}
    except Exception as e:
        return {'error': str(e)}


def check_dns_history(domain):
    """Check DNS history"""
    try:
        answers = dns.resolver.resolve(domain, 'A')
        return {
            'domain': domain,
            'ip_addresses': [str(rdata) for rdata in answers]
        }
    except Exception as e:
        return {'error': str(e)}


def check_ssl_certificate(url):
    """Check SSL/TLS certificate"""
    try:
        hostname = url.replace('https://', '').replace('http://', '').split('/')[0]
        context = ssl.create_default_context()

        with socket.create_connection((hostname, 443)) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()

        # Get more details with OpenSSL
        cert_openssl = SSL.Connection(SSL.SSLv23_METHOD, socket.socket(socket.AF_INET, socket.SOCK_STREAM))
        cert_openssl.connect((hostname, 443))
        cert_openssl.do_handshake()
        x509 = cert_openssl.get_peer_certificate()

        return {
            'subject': dict(x[0] for x in cert['subject']),
            'issuer': dict(x[0] for x in cert['issuer']),
            'version': cert['version'],
            'serialNumber': cert['serialNumber'],
            'notBefore': cert['notBefore'],
            'notAfter': cert['notAfter'],
            'expires_in_days': (datetime.datetime.strptime(cert['notAfter'],
                                                           '%b %d %H:%M:%S %Y %Z') - datetime.datetime.now()).days,
            'signature_algorithm': x509.get_signature_algorithm().decode('utf-8'),
            'bits': x509.get_pubkey().bits()
        }
    except Exception as e:
        return {'error': str(e)}


# Credential Protection Functions
def check_password_breach(password):
    """Check if password has been exposed (using k-anonymity)"""
    try:
        # Hash the password with SHA-1
        sha1_password = hashlib.sha1(password.encode('utf-8')).hexdigest().upper()
        prefix = sha1_password[:5]

        # Check against HIBP
        response = requests.get(f'https://api.pwnedpasswords.com/range/{prefix}')
        if response.status_code == 200:
            hashes = response.text.splitlines()
            for h in hashes:
                if sha1_password[5:] in h:
                    count = int(h.split(':')[1])
                    return {
                        'is_compromised': True,
                        'times_exposed': count
                    }
            return {'is_compromised': False}
        return {'error': 'API request failed'}
    except Exception as e:
        return {'error': str(e)}


# Advanced Threat Detection Functions
def generate_threat_report(indicator):
    """Generate threat intelligence report using ThreatMiner API"""
    try:
        # First determine if the indicator is IP, domain, or hash
        indicator_type = None
        if '.' in indicator and len(indicator.split('.')) >= 2:
            indicator_type = 'domain'
        elif indicator.count(':') >= 2:  # IPv6
            indicator_type = 'ip'
        elif indicator.replace('.', '').isdigit():  # IPv4
            indicator_type = 'ip'
        elif len(indicator) in [32, 40, 64]:  # MD5, SHA-1, SHA-256
            indicator_type = 'hash'

        if not indicator_type:
            return {'error': 'Could not determine indicator type'}

        base_url = 'https://api.threatminer.org/v2'
        endpoints = {
            'ip': f'{base_url}/host.php?q={indicator}&rt=1',
            'domain': f'{base_url}/domain.php?q={indicator}&rt=1',
            'hash': f'{base_url}/sample.php?q={indicator}&rt=1'
        }

        response = requests.get(endpoints[indicator_type])

        if response.status_code == 200:
            data = response.json()

            # Process the data into a structured report
            report = {
                'indicator': indicator,
                'type': indicator_type,
                'sources': [],
                'related_indicators': [],
                'malware_families': [],
                'tags': []
            }

            # Parse different response types
            if indicator_type == 'ip':
                if data.get('results'):
                    for result in data['results']:
                        if result.get('asn'):
                            report['asn'] = result['asn']
                        if result.get('country'):
                            report['country'] = result['country']
                        if result.get('malware'):
                            report['malware_families'].extend(result['malware'])
                        if result.get('sources'):
                            report['sources'].extend(result['sources'])

            elif indicator_type == 'domain':
                if data.get('results'):
                    for result in data['results']:
                        if result.get('ip'):
                            report['related_indicators'].append({
                                'type': 'ip',
                                'value': result['ip']
                            })
                        if result.get('malware'):
                            report['malware_families'].extend(result['malware'])

            elif indicator_type == 'hash':
                if data.get('results'):
                    for result in data['results']:
                        if result.get('md5'):
                            report['hash_md5'] = result['md5']
                        if result.get('sha1'):
                            report['hash_sha1'] = result['sha1']
                        if result.get('sha256'):
                            report['hash_sha256'] = result['sha256']
                        if result.get('av_detections'):
                            report['av_detections'] = result['av_detections']
                        if result.get('tags'):
                            report['tags'].extend(result['tags'])

            return report
        else:
            return {'error': f"API request failed with status {response.status_code}"}
    except Exception as e:
        return {'error': str(e)}


def search_cve(software):
    try:
        response = requests.get(f'https://cve.circl.lu/api/search/{software}')
        return response.json()
    except Exception as e:
        return {'error': str(e)}


def get_ip_geolocation(ip_address):
    try:
        response = requests.get(f'http://ip-api.com/json/{ip_address}')
        data = response.json()
        if data.get('isp', '').lower() in ['vpn', 'proxy'] or data.get('org', '').lower() in ['vpn', 'proxy']:
            data['is_proxy'] = True
        else:
            data['is_proxy'] = False
        return data
    except Exception as e:
        return {'error': str(e)}


if __name__ == '__main__':
    app.run(debug=True)