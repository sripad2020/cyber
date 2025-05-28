import joblib
from flask import Flask, request, render_template, jsonify, redirect, url_for
import pandas as pd
import requests
import whois
import datetime
import hashlib
import shodan
import dns.resolver
import ssl
from cachetools import TTLCache
from urllib.parse import urlparse
import logging
import json
from datetime import datetime
from typing import Dict, Any
import icmplib
import socket
from OpenSSL import SSL
from werkzeug.utils import secure_filename
from gvm.connections import UnixSocketConnection
# Advanced Threat Detection Routes
import asyncio
import uuid
from vulnerability_assesment import perform_vulnerability_assessment,make_json_serializable
from gvm.protocols.gmp import Gmp
from gvm.transforms import EtreeTransform
import uuid
import time,re
import google.generativeai as genai
from nltk import sent_tokenize, word_tokenize, FreqDist
from nltk.corpus import stopwords
from pythonping import ping
import matplotlib.pyplot as plt
import dns.resolver
import asyncio
import socket
import aiohttp
import nmap
import json
import os
import ssl
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional
from OpenSSL import crypto
from concurrent.futures import ThreadPoolExecutor
from vulners import Vulners
from gvm.connections import UnixSocketConnection
from gvm.transforms import EtreeTransform
from gvm.protocols.gmp import Gmp
import socket
import ssl
import datetime
import requests
from urllib.parse import urlparse

import os



app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['SCAN_STORAGE'] = os.path.join(os.getcwd(), 'scan_results')
app.config['SECRET_KEY'] = 'your-secret-key-here'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SCAN_STORAGE'], exist_ok=True)
os.makedirs('static/shap_plots', exist_ok=True)


VT_API_KEY = '99e4922915b2a1c753dfd66e541d41df6a3522cb906b6c0d6ae7c1df6f529ae5'
SHODAN_API_KEY = 'HrwMvhaNOf8SvpLYMazaYl2Fv6E3iFQe'
ABUSEIPDB_API_KEY = '8d727504a8cb301f17446f78bd41dfa8b2c53e029c1de79fb2f8ca9b1cd3f7e444f42f87ab7ee058'
SAFE_BROWSING_API_KEY = 'AIzaSyDyroTCmCRaundFKZL4z6rBoEnKCSIa9Qk'

# Initialize Shodan API
shodan_api = shodan.Shodan(SHODAN_API_KEY)

plot_dir = 'static/shap_plots'
os.makedirs(plot_dir, exist_ok=True)


@app.route('/')
def info():
    return render_template('information.html')

@app.route('/inputs')
def indexed():
    return render_template("inputs.html")
def convert_paragraph_to_points(paragraph, num_points=5):
    sentences = sent_tokenize(paragraph)
    words = word_tokenize(paragraph.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    freq_dist = FreqDist(filtered_words)
    sentence_scores = {}
    for sentence in sentences:
        sentence_word_tokens = word_tokenize(sentence.lower())
        sentence_word_tokens = [word for word in sentence_word_tokens if word.isalnum()]
        score = sum(freq_dist.get(word, 0) for word in sentence_word_tokens)
        sentence_scores[sentence] = score
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    key_points = sorted_sentences[:num_points]
    return key_points

def clean_text(text):
    return re.sub(r'\*\*|\*', '', text)

def clean_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'`{3}.*?`{3}', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'^\s*>+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\*\-+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()
@app.route('/process', methods=['POST'])
def process():
    try:
        # 1. Retrieve and validate form inputs
        required_fields = [
            'Fwd Pkt Len Min',
            'Bwd Pkt Len Min',
            'Flow IAT Min',
            'Pkt Len Min'
        ]

        # Check for missing fields
        missing = [field for field in required_fields if field not in request.form]
        if missing:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(missing)}'
            }), 400

        # Convert inputs to floats
        try:
            features = {
                'Fwd Pkt Len Min': float(request.form['Fwd Pkt Len Min']),
                'Bwd Pkt Len Min': float(request.form['Bwd Pkt Len Min']),
                'Flow IAT Min': float(request.form['Flow IAT Min']),
                'Pkt Len Min': float(request.form['Pkt Len Min'])
            }
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'Invalid input. All values must be numbers.'
            }), 400

        # 2. Make prediction
        try:
            model = joblib.load('the_light.pkl')
            input_data = [[features[field] for field in required_fields]]
            prediction = model.predict(input_data)[0]
            pred_proba = model.predict_proba(input_data)[0][prediction]

            # Map numeric prediction to label
            attack_types = {
                0: "Brute_Force",
                1: "HTTP_DDoS",
                2: "ICMP_Flood",
                3: "Normal",
                4: "Port_Scan",
                5: "Web_Crawling"
            }
            prediction_label = attack_types.get(prediction, "Unknown")

        except FileNotFoundError:
            return jsonify({
                'status': 'error',
                'message': 'Model file not found.'
            }), 500
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Prediction failed: {str(e)}'
            }), 500

        # 3. Generate Gemini explanation
        try:
            genai.configure(api_key='AIzaSyAN5rU9-qHNGFz2ChZh_LIwwybqNEXr7tI')
            model = genai.GenerativeModel('gemini-1.5-flash')

            prompt = f"""
            Analyze these network traffic features and explain why they might indicate a {prediction_label} attack:

            - Forward Packet Length Min: {features['Fwd Pkt Len Min']}
            - Backward Packet Length Min: {features['Bwd Pkt Len Min']}
            - Flow Inter-Arrival Time Min: {features['Flow IAT Min']}
            - Packet Length Min: {features['Pkt Len Min']}

            Provide:
            1. A simple explanation of what this attack type means
            2. How these specific values suggest this attack
            3. Recommended mitigation steps
            4. Potential false positives to consider

            Format the response in clear bullet points.
            """

            response = model.generate_content(prompt)
            explanation = clean_markdown(response.text)
            explanation_points = [line.strip() for line in explanation.split('\n') if line.strip()]

        except Exception as e:
            explanation_points = [f"Could not generate explanation: {str(e)}"]


        return jsonify({
            'status': 'success',
            'prediction': prediction_label,
            'features': features,
            'explanation': explanation_points,
            'technical_details': {
                'model_used': 'LGBMClassifier',
                'feature_importance': {
                    'Fwd Pkt Len Min': 'High',
                    'Bwd Pkt Len Min': 'Medium',
                    'Flow IAT Min': 'High',
                    'Pkt Len Min': 'Medium'
                },
                'confidence': f"{pred_proba:.1%}"
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        }), 500


@app.route('/phishing', methods=['GET', 'POST'])
def phishing():
    results = {}
    if request.method == 'POST':
        if 'phishing-url' in request.form and request.form['phishing-url']:
            url = request.form['phishing-url']
            results['phishing_url'] = check_phishing_url(url)
            print(results)
        if 'email-content' in request.form and request.form['email-content']:
            email_content = request.form['email-content']
            results['email_analysis'] = analyze_email_content(email_content)
        if 'domain-age' in request.form and request.form['domain-age']:
            domain = request.form['domain-age']
            results['domain_age'] = check_domain_age(domain)
    return render_template('phishing.html', results=results)

# Malware Analysis Routes
@app.route('/malware', methods=['GET', 'POST'])
def malware():
    results = {}
    if request.method == 'POST':
        if 'file-upload' in request.files and request.files['file-upload'].filename != '':
            uploaded_file = request.files['file-upload']
            results['file_scan'] = scan_uploaded_file(uploaded_file)
        if 'file-hash' in request.form and request.form['file-hash']:
            file_hash = request.form['file-hash']
            results['hash_lookup'] = check_file_hash(file_hash)
    return render_template('malware.html', results=results)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache for WHOIS results (TTL: 1 hour)
whois_cache = TTLCache(maxsize=100, ttl=3600)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for non-serializable objects."""
    def default(self, obj):
        logger.debug(f"Encoding object of type: {type(obj)}")
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif isinstance(obj, datetime):
            logger.debug(f"Converting datetime: {obj}")
            return obj.isoformat()
        elif isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        logger.debug(f"Converting unhandled type {type(obj)} to string: {obj}")
        return str(obj)

def make_json_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable formats using CustomJSONEncoder."""
    logger.debug(f"Serializing object: {obj}")
    serialized = json.loads(json.dumps(obj, cls=CustomJSONEncoder))
    logger.debug(f"Serialized result: {serialized}")
    return serialized

def validate_domain(domain: str) -> bool:
    """Validate a domain name."""
    domain_pattern = r'^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$'
    return bool(re.match(domain_pattern, domain))

def validate_host(host: str) -> bool:
    """Validate a host (IP or domain)."""
    ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    return bool(re.match(ip_pattern, host)) or validate_domain(host)

def validate_url(url: str) -> bool:
    """Validate a URL."""
    url_pattern = r'^(https?:\/\/)?([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:\/.*)?$'
    return bool(re.match(url_pattern, url))

async def whois_lookup(domain: str) -> Dict[str, Any]:
    """Perform a WHOIS lookup for a domain."""
    if not validate_domain(domain):
        return {'domain': domain, 'error': 'Invalid domain name'}

    try:
        # Check cache first
        if domain in whois_cache:
            logger.info(f"Returning cached WHOIS for {domain}")
            return whois_cache[domain]

        async with aiohttp.ClientSession() as session:
            w = whois.whois(domain)
            result = {
                'domain': domain,
                'registrar': w.get('registrar', 'N/A'),
                'creation_date': w.get('creation_date', 'N/A'),
                'expiration_date': w.get('expiration_date', 'N/A'),
                'name_servers': w.get('name_servers', []),
                'registrant': w.get('registrant', 'N/A'),
                'contacts': {
                    'email': w.get('email', 'N/A'),
                    'phone': w.get('phone', 'N/A'),
                    'organization': w.get('org', 'N/A')
                },
                'status': w.get('status', [])
            }
            whois_cache[domain] = result
            return result
    except Exception as e:
        logger.error(f"WHOIS lookup failed for {domain}: {str(e)}")
        return {'domain': domain, 'error': f'WHOIS lookup failed: {str(e)}'}

async def latency_test(host: str) -> Dict[str, Any]:
    """Perform a network latency test using ICMP ping."""
    if not validate_host(host):
        return {'host': host, 'error': 'Invalid host (IP or domain)'}

    try:
        # Use icmplib for ICMP ping (requires admin privileges on Windows)
        results = await asyncio.to_thread(
            icmplib.ping, host, count=5, interval=0.5, timeout=2
        )
        if not results.is_alive:
            return {'host': host, 'error': 'Host unreachable'}

        raw_rtts = results.rtts if results.rtts else []
        logger.info(f"Raw RTTs for {host}: {raw_rtts}")
        result = {
            'host': host,
            'is_alive': results.is_alive,
            'packet_loss': results.packet_loss,
            'min_rtt': results.min_rtt,
            'max_rtt': results.max_rtt,
            'avg_rtt': results.avg_rtt,
            'raw_rtts': raw_rtts
        }
        logger.info(f"Latency test result for {host}: {result}")
        return result
    except icmplib.exceptions.NameLookupError:
        logger.error(f"Name lookup failed for {host}")
        return {'host': host, 'error': 'Host name resolution failed'}
    except icmplib.exceptions.SocketPermissionError:
        logger.error(f"Permission error for ping on {host}")
        return {'host': host, 'error': 'Ping requires admin privileges'}
    except Exception as e:
        logger.error(f"Latency test failed for {host}: {str(e)}")
        return {'host': host, 'error': f'Latency test failed: {str(e)}'}

async def security_headers_analysis(url: str) -> Dict[str, Any]:
    """Analyze HTTP security headers of a website."""
    if not validate_url(url):
        return {'url': url, 'error': 'Invalid URL'}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                headers = dict(response.headers)
                recommended_headers = {
                    'Content-Security-Policy': 'Prevents XSS and data injection attacks',
                    'X-Frame-Options': 'Protects against clickjacking',
                    'X-Content-Type-Options': 'Prevents MIME-type sniffing',
                    'Strict-Transport-Security': 'Enforces HTTPS',
                    'Referrer-Policy': 'Controls referrer information',
                    'Permissions-Policy': 'Controls browser features'
                }
                result = {
                    'url': url,
                    'status_code': response.status,
                    'headers': headers,
                    'missing_headers': [],
                    'issues': []
                }
                for header, desc in recommended_headers.items():
                    if header not in headers:
                        result['missing_headers'].append(header)
                        result['issues'].append(f'Missing {header}: {desc}')
                    elif header == 'Strict-Transport-Security' and 'max-age=0' in headers[header]:
                        result['issues'].append('HSTS max-age is 0, rendering it ineffective')
                return result
    except aiohttp.ClientError as e:
        logger.error(f"Security headers analysis failed for {url}: {str(e)}")
        return {'url': url, 'error': f'HTTP request failed: {str(e)}'}
    except Exception as e:
        logger.error(f"Security headers analysis failed for {url}: {str(e)}")
        return {'url': url, 'error': f'Analysis failed: {str(e)}'}

@app.route('/network', methods=['GET', 'POST'])
async def network():
    results = {}
    if request.method == 'POST':
        # WHOIS Lookup
        if 'whois-lookup' in request.form and request.form['whois-lookup']:
            domain = request.form['whois-lookup'].strip()
            results['whois_lookup'] = await whois_lookup(domain)
        # Latency Test
        elif 'latency-test' in request.form and request.form['latency-test']:
            host = request.form['latency-test'].strip()
            results['latency_test'] = await latency_test(host)
        # Security Headers Analysis
        elif 'headers-analysis' in request.form and request.form['headers-analysis']:
            url = request.form['headers-analysis'].strip()
            results['security_headers'] = await security_headers_analysis(url)

    # Ensure results are JSON-serializable for Jinja2
    logger.info("Results before serialization: %s", results)
    results = make_json_serializable(results)
    logger.info("Results after serialization: %s", results)
    return render_template('network.html', results=results)


def basic_port_scan(ip_address):
    try:
        # Common ports to scan
        common_ports = {
            21: 'FTP', 22: 'SSH', 23: 'Telnet', 25: 'SMTP',
            80: 'HTTP', 443: 'HTTPS', 3306: 'MySQL', 3389: 'RDP'
        }
        open_ports = []
        socket.setdefaulttimeout(1)  # 1-second timeout
        for port, service in common_ports.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((ip_address, port))
            if result == 0:
                open_ports.append({'port': port, 'service': service, 'status': 'Open'})
            sock.close()
        return {
            'ip': ip_address,
            'open_ports': open_ports,
            'status': 'Completed' if open_ports else 'No open ports found'
        }
    except Exception as e:
        return {'error': f'Port scan failed: {str(e)}'}


def dns_lookup(domain):
    try:
        resolver = dns.resolver.Resolver()
        resolver.nameservers = ['8.8.8.8']  # Use Google's DNS
        records = {'A': [], 'MX': [], 'NS': []}
        # A records
        try:
            answers = resolver.resolve(domain, 'A')
            records['A'] = [str(rdata) for rdata in answers]
        except Exception:
            records['A'] = ['None']
        # MX records
        try:
            answers = resolver.resolve(domain, 'MX')
            records['MX'] = [str(rdata.exchange) for rdata in answers]
        except Exception:
            records['MX'] = ['None']
        # NS records
        try:
            answers = resolver.resolve(domain, 'NS')
            records['NS'] = [str(rdata.target) for rdata in answers]
        except Exception:
            records['NS'] = ['None']
        return {
            'domain': domain,
            'records': records,
            'status': 'Completed'
        }
    except Exception as e:
        return {'error': f'DNS lookup failed: {str(e)}'}


def ssl_version_check(url):
    try:
        # Clean URL (remove http:// or https://)
        hostname = url.replace('https://', '').replace('http://', '').split('/')[0]
        port = 443
        supported_versions = []
        version_names = {
            'PROTOCOL_TLSv1': 'TLS 1.0',
            'PROTOCOL_TLSv1_1': 'TLS 1.1',
            'PROTOCOL_TLSv1_2': 'TLS 1.2',
            'PROTOCOL_TLS': 'TLS 1.3'  # PROTOCOL_TLS is used for 1.3 in some implementations
        }

        # Test each version
        for version_name, display_name in version_names.items():
            try:
                if hasattr(ssl, version_name):
                    context = ssl.SSLContext(getattr(ssl, version_name))
                    context.set_ciphers('ALL:@SECLEVEL=0')
                    with socket.create_connection((hostname, port), timeout=5) as sock:
                        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                            supported_versions.append(display_name)
            except Exception:
                continue

        return {
            'hostname': hostname,
            'supported_versions': supported_versions,
            'status': 'Completed' if supported_versions else 'No TLS versions supported'
        }
    except Exception as e:
        return {'error': f'SSL check failed: {str(e)}'}


def http_header_analysis(url):
    try:
        # Ensure URL starts with http:// or https://
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        response = requests.get(url, timeout=5, allow_redirects=True)
        headers = response.headers
        security_headers = {
            'X-Frame-Options': 'Prevents clickjacking',
            'Content-Security-Policy': 'Controls resource loading',
            'Strict-Transport-Security': 'Enforces HTTPS',
            'X-Content-Type-Options': 'Prevents MIME sniffing'
        }
        header_results = {}
        for header, desc in security_headers.items():
            header_results[header] = {
                'value': headers.get(header, 'Not set'),
                'description': desc
            }
        return {
            'url': url,
            'headers': header_results,
            'server': headers.get('Server', 'Unknown'),
            'status': 'Completed'
        }
    except Exception as e:
        return {'error': f'Header analysis failed: {str(e)}'}


def ping_test(target):
    try:
        # Resolve hostname to IP if needed
        try:
            ip = socket.gethostbyname(target)
        except socket.gaierror:
            return {'error': f'Could not resolve {target}'}
        # Perform 3 pings
        latencies = []
        for _ in range(3):
            try:
                response = ping(target, count=1, timeout=2)
                if response.success():
                    latencies.append(response.rtt_avg_ms)
            except Exception:
                continue
            time.sleep(0.5)
        if not latencies:
            return {
                'target': target,
                'ip': ip,
                'status': 'Host unreachable',
                'average_latency': 'N/A'
            }
        avg_latency = sum(latencies) / len(latencies)
        return {
            'target': target,
            'ip': ip,
            'status': 'Host reachable',
            'average_latency': f'{avg_latency:.2f} ms'
        }
    except Exception as e:
        return {'error': f'Ping test failed: {str(e)}'}


@app.route('/progress/<scan_id>', methods=['GET'])
def check_progress(scan_id):
    result_file = os.path.join(app.config['SCAN_STORAGE'], f'{scan_id}.json')
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            result = json.load(f)
        return jsonify({'status': 'completed', 'result': result})
    return jsonify({'status': 'running'})

# Credential Protection Routes
@app.route('/credentials', methods=['GET', 'POST'])
def credentials():
    results = {}
    if request.method == 'POST':
        if 'email-check' in request.form and request.form['email-check']:
            email = request.form['email-check']
            results['email_breach'] = check_email_breach(email)
        if 'password-check' in request.form and request.form['password-check']:
            password = request.form['password-check']
            results['password_breach'] = check_password_breach(password)
    return render_template('credentials.html', results=results)




# Placeholder for scan_ports (used in generate_threat_report)
def scan_ports(ip):
    try:
        shodan_api = shodan.Shodan(SHODAN_API_KEY)
        host_info = shodan_api.host(ip)
        return {
            'matches': len(host_info.get('data', [])),
            'ports': list(set([entry['port'] for entry in host_info.get('data', [])])),
            'services': list(set([entry.get('product', '') for entry in host_info.get('data', []) if entry.get('product')])),
            'os': [host_info.get('os', '')] if host_info.get('os') else [],
            'vulnerabilities': host_info.get('vulns', []),
            'banners': [entry.get('data', '')[:100] for entry in host_info.get('data', []) if entry.get('data')],
            'last_seen': host_info.get('last_update'),
            'organizations': [host_info.get('org', '')] if host_info.get('org') else [],
            'geolocations': [host_info.get('country_name', '')] if host_info.get('country_name') else [],
            'asn': [host_info.get('asn', '')] if host_info.get('asn') else [],
            'isp': [host_info.get('isp', '')] if host_info.get('isp') else [],
            'domains': host_info.get('hostnames', [])
        }
    except shodan.APIError as e:
        return {'error': f"Shodan API error: {str(e)}", 'error_type': 'ShodanAPIError'}

def get_geolocation(ip: str) -> dict:
    """Fetch geolocation data for an IP address using ip-api.com."""
    try:
        response = requests.get(f'http://ip-api.com/json/{ip}', timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('status') == 'success':
            return {
                'ip': ip,
                'city': data.get('city', 'N/A'),
                'country': data.get('country', 'N/A'),
                'region': data.get('regionName', 'N/A'),
                'latitude': data.get('lat', 'N/A'),
                'longitude': data.get('lon', 'N/A'),
                'isp': data.get('isp', 'N/A'),
                'org': data.get('org', 'N/A'),
                'error': None
            }
        else:
            return {'ip': ip, 'error': f"Geolocation lookup failed: {data.get('message', 'Unknown error')}"}
    except requests.RequestException as e:
        return {'ip': ip, 'error': f"Geolocation request failed: {str(e)}"}

@app.route('/advanced', methods=['GET', 'POST'])
def advanced():
    results = {}
    if request.method == 'POST':
        if 'vuln-assessment' in request.form and request.form['vuln-assessment']:
            target = request.form['vuln-assessment']
            scan_id = str(uuid.uuid4())
            # Run async function in Flask        context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results['vuln_assessment'] = loop.run_until_complete(
                    perform_vulnerability_assessment(target, scan_id)
                )
            finally:
                loop.close()
        elif 'geolocation-ip' in request.form and request.form['geolocation-ip']:
            ip = request.form['geolocation-ip']
            results['geolocation'] = get_geolocation(ip)

    # Ensure results are JSON-serializable for Jinja2
    results = make_json_serializable(results)
    print("Results being passed to template:", results)
    return render_template('advanced.html', results=results)

def check_phishing_url(url):
    # Google Safe Browsing API
    endpoint = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
    headers = {"Content-Type": "application/json"}

    payload = {
        "client": {
            "clientId": "your-security-app",
            "clientVersion": "1.0"
        },
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }

    result = {
        "url": url,
        "status": "Safe",
        "threat_type": None,
        "severity": None,
        "last_seen": None,
        "cache_duration": None,
        "raw_response": None,
        "http_status": None,
        "final_url": None,
        "content_length": None,
        "headers": None,
        "response_time_ms": None,
        "domain": urlparse(url).netloc if urlparse(url).netloc else "N/A",
        "error": None,
        "error_type": None,
        "error_message": None
    }

    # Google Safe Browsing API request
    try:
        response = requests.post(
            f"{endpoint}?key={SAFE_BROWSING_API_KEY}",
            json=payload,
            headers=headers
        )
        response.raise_for_status()

        data = response.json()
        result["raw_response"] = data

        if "matches" in data:
            result["status"] = "Malicious"
            result["threat_type"] = [match["threatType"] for match in data["matches"]]
            first_match = data["matches"][0]
            result.update({
                "severity": first_match.get("threatEntryMetadata", {}).get("severity", "UNKNOWN"),
                "last_seen": first_match.get("threatEntryMetadata", {}).get("lastSeen", None),
                "cache_duration": data.get("cacheDuration", "0s")
            })

    except requests.exceptions.HTTPError as e:
        result.update({
            "status": "Error",
            "error_type": "HTTP Error (Safe Browsing API)",
            "error_message": str(e),
            "status_code": e.response.status_code if e.response else None
        })
        return result
    except Exception as e:
        result.update({
            "status": "Error",
            "error_type": f"Safe Browsing API Error: {type(e).__name__}",
            "error_message": str(e)
        })
        return result
    try:
        start_time = time.time()
        response = requests.get(
            url,
            allow_redirects=True,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        response_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds

        result.update({
            "http_status": response.status_code,
            "final_url": response.url,
            "content_length": len(response.content) if response.content else None,
            "headers": dict(response.headers),
            "response_time_ms": round(response_time_ms, 2)
        })

    except requests.exceptions.RequestException as e:
        result.update({
            "error": "HTTP Request Failed",
            "error_type": f"HTTP Error: {type(e).__name__}",
            "error_message": str(e)
        })
    return result
def analyze_email_content(content):
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
        """
        Enhanced domain information checker that returns:
        - WHOIS information (creation date, registrar, expiration)
        - DNS records (A, MX, TXT, NS)
        - SSL certificate details
        - Domain reputation indicators
        - Basic connectivity check
        """
        result = {
            'domain': domain,
            'whois': {},
            'dns': {},
            'ssl': {},
            'reputation': {},
            'connectivity': {},
            'error': None
        }

        try:
            # Clean domain (remove http/https/www)
            domain = urlparse(domain).netloc if urlparse(domain).netloc else domain
            domain = domain.replace('www.', '')

            # WHOIS Information
            try:
                domain_info = whois.whois(domain)
                creation_date = domain_info.creation_date
                expiration_date = domain_info.expiration_date

                # Handle cases where dates are lists
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                if isinstance(expiration_date, list):
                    expiration_date = expiration_date[0]

                age_days = (datetime.datetime.now() - creation_date).days if creation_date else None

                result['whois'] = {
                    'creation_date': str(creation_date),
                    'expiration_date': str(expiration_date),
                    'age_days': age_days,
                    'registrar': domain_info.registrar,
                    'name_servers': domain_info.name_servers,
                    'status': domain_info.status,
                    'emails': domain_info.emails,
                    'is_new': age_days is not None and age_days < 30,
                    'is_expired': expiration_date and expiration_date < datetime.datetime.now()
                }
            except Exception as e:
                result['whois']['error'] = str(e)

            # DNS Records
            resolver = dns.resolver.Resolver()
            resolver.timeout = 5
            resolver.lifetime = 5

            dns_types = ['A', 'MX', 'TXT', 'NS', 'CNAME']
            for record_type in dns_types:
                try:
                    answers = resolver.resolve(domain, record_type)
                    if record_type == 'MX':
                        result['dns'][record_type] = [str(r.exchange) for r in answers]
                    else:
                        result['dns'][record_type] = [str(r) for r in answers]
                except:
                    result['dns'][record_type] = []

            # SSL Certificate Information
            try:
                context = ssl.create_default_context()
                with socket.create_connection((domain, 443), timeout=5) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        cert = ssock.getpeercert()

                        # Parse certificate dates
                        not_before = datetime.datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
                        not_after = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        cert_age = (datetime.datetime.now() - not_before).days

                        result['ssl'] = {
                            'issued_to': dict(x[0] for x in cert['subject']),
                            'issuer': dict(x[0] for x in cert['issuer']),
                            'valid_from': str(not_before),
                            'valid_to': str(not_after),
                            'cert_age_days': cert_age,
                            'is_valid': datetime.datetime.now() < not_after,
                            'is_new': cert_age < 30,
                            'serial_number': cert.get('serialNumber'),
                            'version': cert.get('version')
                        }
            except Exception as e:
                result['ssl']['error'] = str(e)

            # Basic Connectivity Check
            try:
                response = requests.get(f"http://{domain}", timeout=5, allow_redirects=True)
                result['connectivity'] = {
                    'http_status': response.status_code,
                    'final_url': response.url,
                    'response_time_ms': response.elapsed.total_seconds() * 1000,
                    'headers': dict(response.headers),
                    'content_length': len(response.content)
                }
            except Exception as e:
                result['connectivity']['error'] = str(e)

            # Reputation Indicators
            result['reputation'] = {
                'has_mx_records': len(result['dns'].get('MX', [])) > 0,
                'has_spf_record': any('spf' in txt.lower() for txt in result['dns'].get('TXT', [])),
                'has_dmarc_record': any('dmarc' in txt.lower() for txt in result['dns'].get('TXT', [])),
                'is_https_working': 'error' not in result['ssl'],
                'is_redirecting': result['connectivity'].get('final_url', '').lower() != f"http://{domain.lower()}",
                'common_nameserver': any(ns.endswith(('.com', '.net')) for ns in result['dns'].get('NS', []))
            }

        except Exception as e:
            result['error'] = str(e)
        return result

# Malware Analysis Functions
import tempfile
import os
import hashlib
import requests
from werkzeug.utils import secure_filename

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scan_uploaded_file(file):
    try:
        # Use NamedTemporaryFile for better file management
        with tempfile.NamedTemporaryFile(delete=False, suffix=secure_filename(file.filename)) as temp_file:
            # Save uploaded file to temporary file
            file.save(temp_file.name)
            filepath = temp_file.name

            # Calculate hashes
            with open(filepath, 'rb') as f:
                file_content = f.read()
                sha256_hash = hashlib.sha256(file_content).hexdigest()
                sha1_hash = hashlib.sha1(file_content).hexdigest()
                md5_hash = hashlib.md5(file_content).hexdigest()

            # Get file size
            file_size = os.path.getsize(filepath)

            # Initialize result dictionary
            result = {
                'filename': secure_filename(file.filename),
                'sha256': sha256_hash,
                'sha1': sha1_hash,
                'md5': md5_hash,
                'file_size_bytes': file_size,
                'file_type': None,
                'status': 'Pending',
                'positives': 0,
                'total': 0,
                'scan_date': None,
                'permalink': None,
                'scans': None,
                'first_submission': None,
                'last_submission': None,
                'times_submitted': None,
                'error': None,
                'error_type': None,
                'error_message': None
            }

            # Check if report exists
            vt_report = check_file_hash(sha256_hash)
            if vt_report.get('response_code') == 1:
                result.update({
                    'status': 'Completed',
                    'positives': vt_report.get('positives', 0),
                    'total': vt_report.get('total', 0),
                    'scan_date': vt_report.get('scan_date'),
                    'permalink': vt_report.get('permalink'),
                    'scans': vt_report.get('scans'),
                    'file_type': vt_report.get('type'),
                    'first_submission': vt_report.get('first_seen'),
                    'last_submission': vt_report.get('last_seen'),
                    'times_subm itted': vt_report.get('times_submitted')
                })
                return result

            # No report found, initiate scan
            with open(filepath, 'rb') as f:
                files = {'file': (result['filename'], f)}
                params = {'apikey': VT_API_KEY}
                logger.info(f"Initiating VirusTotal scan for {result['filename']}")
                scan_response = requests.post(
                    'https://www.virustotal.com/vtapi/v2/file/scan',
                    files=files,
                    params=params
                )
                scan_response.raise_for_status()
                scan_data = scan_response.json()

            # Wait briefly and attempt to retrieve the report
            time.sleep(5)
            report_response = requests.get(
                'https://www.virustotal.com/vtapi/v2/file/report',
                params={'apikey': VT_API_KEY, 'resource': scan_data.get('scan_id')}
            )
            report_response.raise_for_status()
            report_data = report_response.json()

            if report_data.get('response_code') == 1:
                result.update({
                    'status': 'Completed',
                    'positives': report_data.get('positives', 0),
                    'total': report_data.get('total', 0),
                    'scan_date': report_data.get('scan_date'),
                    'permalink': report_data.get('permalink'),
                    'scans': report_data.get('scans'),
                    'file_type': report_data.get('type'),
                    'first_submission': report_data.get('first_seen'),
                    'last_submission': report_data.get('last_seen'),
                    'times_submitted': report_data.get('times_submitted')
                })
            else:
                result.update({
                    'status': 'Initiated',
                    'scan_id': scan_data.get('scan_id'),
                    'permalink': scan_data.get('permalink')
                })

            return result

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error during scan for {secure_filename(file.filename)}: {str(e)}")
        return {
            'filename': secure_filename(file.filename),
            'status': 'Error',
            'error': 'HTTP Error',
            'error_type': str(type(e).__name__),
            'error_message': str(e),
            'status_code': e.response.status_code if e.response else None
        }
    except PermissionError as e:
        logger.error(f"Permission error during scan for {secure_filename(file.filename)}: {str(e)}")
        return {
            'filename': secure_filename(file.filename),
            'status': 'Error',
            'error': 'Permission Error',
            'error_type': str(type(e).__name__),
            'error_message': 'File is locked by another process. Please try again.'
        }
    except Exception as e:
        logger.error(f"General error during scan for {secure_filename(file.filename)}: {str(e)}")
        return {
            'filename': secure_filename(file.filename),
            'status': 'Error',
            'error': 'Scan Failed',
            'error_type': str(type(e).__name__),
            'error_message': str(e)
        }
    finally:
        # Ensure temporary file is deleted
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.unlink(filepath)
                logger.info(f"Cleaned up temporary file: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {filepath}: {str(e)}")
def check_file_hash(file_hash):
    try:
        params = {'apikey': VT_API_KEY, 'resource': file_hash}
        response = requests.get(
            'https://www.virustotal.com/vtapi/v2/file/report',
            params=params
        )
        return response.json()
    except Exception as e:
        return {'error': str(e)}

# Network Threat Detection Functions
def check_ip_reputation(ip_address):
    try:
        headers = {'Key': ABUSEIPDB_API_KEY, 'Accept': 'application/json'}
        params = {'ipAddress': ip_address, 'maxAgeInDays': '90'}
        response = requests.get(
            'https://api.abuseipdb.com/api/v2/check',
            headers=headers,
            params=params
        )
        return response.json()
    except Exception as e:
        return {'error': str(e)}

def scan_ports(ip_address):
    try:
        host = shodan_api.host(ip_address)
        vulns = []
        if 'vulns' in host:
            for cve in host['vulns']:
                vulns.append({
                    'cve': cve,
                    'summary': host['vulns'][cve].get('summary', ''),
                    'cvss': host['vulns'][cve].get('cvss', None)
                })
        services = []
        for item in host.get('data', []):
            service = {
                'port': item['port'],
                'transport': item.get('transport', 'tcp'),
                'product': item.get('product', ''),
                'version': item.get('version', ''),
                'banner': item.get('data', '')
            }
            services.append(service)
        return {
            'ip': host['ip_str'],
            'ports': host.get('ports', []),
            'vulnerabilities': vulns,
            'services': services,
            'hostnames': host.get('hostnames', []),
            'country': host.get('country_name', ''),
            'os': host.get('os', ''),
            'last_update': host.get('last_update', ''),
            'shodan_link': f"https://www.shodan.io/host/{host['ip_str']}"
        }
    except shodan.APIError as e:
        return {'error': f"Shodan API error: {str(e)}"}
    except Exception as e:
        return {'error': str(e)}

def check_dns_history(domain):
    try:
        resolver = dns.resolver.Resolver()
        resolver.nameservers = ['8.8.8.8']  # Google DNS

        records = {
            'A': [],
            'MX': [],
            'NS': [],
            'TXT': []
        }

        # A records
        try:
            answers = resolver.resolve(domain, 'A')
            records['A'] = [str(r) for r in answers]
        except:
            pass

        # MX records
        try:
            answers = resolver.resolve(domain, 'MX')
            records['MX'] = [str(r.exchange) for r in answers]
        except:
            pass

        # NS records
        try:
            answers = resolver.resolve(domain, 'NS')
            records['NS'] = [str(r.target) for r in answers]
        except:
            pass

        # TXT records
        try:
            answers = resolver.resolve(domain, 'TXT')
            records['TXT'] = [str(r) for r in answers]
        except:
            pass

        return {
            'domain': domain,
            'records': records,
            'status': 'Completed'
        }
    except Exception as e:
        return {'error': str(e)}


def check_ssl_certificate(url):
    try:
        hostname = url.replace('https://', '').replace('http://', '').split('/')[0]
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443)) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                cipher = ssock.cipher()
                tls_version = ssock.version()
        conn = SSL.Connection(SSL.SSLv23_METHOD, socket.socket(socket.AF_INET, socket.SOCK_STREAM))
        conn.connect((hostname, 443))
        conn.do_handshake()
        x509 = conn.get_peer_certificate()
        not_after = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
        days_remaining = (not_after - datetime.datetime.now()).days
        return {
            'subject': dict(x[0] for x in cert['subject']),
            'issuer': dict(x[0] for x in cert['issuer']),
            'version': cert['version'],
            'serialNumber': cert['serialNumber'],
            'valid_from': cert['notBefore'],
            'valid_to': cert['notAfter'],
            'days_remaining': days_remaining,
            'tls_version': tls_version,
            'cipher': cipher[0] if cipher else None,
            'key_strength': cipher[1] if cipher else None,
            'openssl_details': {
                'signature_algorithm': x509.get_signature_algorithm().decode('utf-8'),
                'bits': x509.get_pubkey().bits(),
                'has_expired': x509.has_expired()
            }
        }
    except Exception as e:
        return {'error': str(e)}



# Credential Protection Functions
def check_email_breach(email):
    try:
        email_hash = hashlib.sha1(email.encode('utf-8')).hexdigest().upper()
        prefix = email_hash[:5]
        headers = {'User-Agent': 'CyberMind-Threat-Detection'}
        response = requests.get(
            f'https://api.pwnedpasswords.com/range/{prefix}',
            headers=headers
        )
        if response.status_code == 200:
            hashes = response.text.splitlines()
            suffix = email_hash[5:]
            for h in hashes:
                if h.startswith(suffix):
                    count = int(h.split(':')[1])
                    return {'is_compromised': True, 'times_exposed': count}
            return {'is_compromised': False}
        return {'error': 'API request failed'}
    except Exception as e:
        return {'error': str(e)}

def check_password_breach(password):
    try:
        sha1_password = hashlib.sha1(password.encode('utf-8')).hexdigest().upper()
        prefix = sha1_password[:5]
        response = requests.get(f'https://api.pwnedpasswords.com/range/{prefix}')
        if response.status_code == 200:
            hashes = response.text.splitlines()
            for h in hashes:
                if sha1_password[5:] in h:
                    count = int(h.split(':')[1])
                    return {'is_compromised': True, 'times_exposed': count}
            return {'is_compromised': False}
        return {'error': 'API request failed'}
    except Exception as e:
        return {'error': str(e)}

# Advanced Threat Detection Functions
def generate_threat_report(indicator):
    try:
        shodan_data = {}
        try:
            is_domain = '.' in indicator and not indicator.replace('.', '').isdigit()
            if is_domain:
                shodan_results = shodan_api.search(f"hostname:{indicator}")
                if shodan_results['total'] > 0:
                    shodan_data = {
                        'shodan_matches': shodan_results['total'],
                        'ports': list(set([result['port'] for result in shodan_results['matches']])),
                        'last_seen': max(shodan_results['matches'], key=lambda x: x.get('timestamp', ''))['timestamp'] if shodan_results['matches'] else None,
                        'organizations': list(set([result.get('org', '') for result in shodan_results['matches']])),
                        'geolocations': list(set([result.get('location', {}).get('country_name', '') for result in shodan_results['matches']]))
                    }
                else:
                    shodan_data = {'shodan_matches': 0, 'ports': [], 'last_seen': None, 'organizations': [], 'geolocations': []}
                protocol_data = {}
                for protocol in ['http', 'ssh', 'rdp']:
                    protocol_results = shodan_api.search(f"hostname:{indicator} {protocol}")
                    protocol_data[protocol] = {
                        'matches': protocol_results['total'],
                        'ports': list(set([result['port'] for result in protocol_results['matches']])),
                    }
                shodan_data['protocol_specific'] = protocol_data
            else:
                shodan_data = scan_ports(indicator)
                historical_data = shodan_api.host_history(indicator) if hasattr(shodan_api, 'host_history') else {'history': []}
                shodan_data['historical_scans'] = {
                    'scan_count': len(historical_data.get('history', [])),
                    'last_scan': max([h['timestamp'] for h in historical_data.get('history', [])], default=None)
                }
                alerts = shodan_api.alerts() if hasattr(shodan_api, 'alerts') else []
                shodan_data['network_alerts'] = [
                    {
                        'id': alert['id'],
                        'name': alert['name'],
                        'created': alert['created'],
                        'ip_range': alert.get('filters', {}).get('ip', '')
                    } for alert in alerts if indicator in alert.get('filters', {}).get('ip', '')
                ]
        except shodan.APIError as e:
            shodan_data = {'error': f"Shodan API error: {str(e)}"}
        vt_data = {}
        try:
            params = {'apikey': VT_API_KEY, 'resource': indicator}
            response = requests.get('https://www.virustotal.com/vtapi/v2/url/report', params=params)
            vt_data = response.json()
        except Exception as e:
            vt_data = {'error': str(e)}
        return {
            'indicator': indicator,
            'shodan_data': shodan_data,
            'virustotal_data': vt_data
        }
    except Exception as e:
        return {'error': str(e)}

def search_cve(software):
    try:
        response = requests.get(f'https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={software}')
        if response.status_code == 200:
            cve_data = response.json()
            vulnerabilities = []
            for vuln in cve_data.get('vulnerabilities', []):
                cve = vuln.get('cve', {})
                vulnerabilities.append({
                    'id': cve.get('id'),
                    'description': cve.get('descriptions', [{}])[0].get('value'),
                    'published': cve.get('published'),
                    'severity': cve.get('metrics', {}).get('cvssMetricV31', [{}])[0].get('cvssData', {}).get('baseSeverity')
                })
            return {
                'software': software,
                'cve_count': len(vulnerabilities),
                'vulnerabilities': vulnerabilities[:10]
            }
        return {'error': 'API request failed'}
    except Exception as e:
        return {'error': str(e)}

def get_ip_geolocation(ip_address):
    try:
        response = requests.get(f'http://ip-api.com/json/{ip_address}')
        if response.status_code == 200:
            data = response.json()
            return {
                'ip': ip_address,
                'location': {
                    'city': data.get('city'),
                    'country': data.get('country'),
                    'region': data.get('regionName'),
                    'latitude': data.get('lat'),
                    'longitude': data.get('lon')
                },
                'isp': data.get('isp'),
                'org': data.get('org')
            }
        return {'error': 'API request failed'}
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)