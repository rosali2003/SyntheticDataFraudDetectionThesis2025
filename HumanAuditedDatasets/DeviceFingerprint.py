import pandas as pd
import numpy as np

def generate_device_fingerprints_dataset(n_samples=50):

    browsers = ['Chrome', 'Firefox', 'Safari', 'Edge']
    os_types = ['Windows', 'MacOS', 'iOS', 'Android']
    screen_resolutions = ['1920x1080', '1366x768', '2560x1440', '375x812']

    os_browser_compatibility = {
        'Windows': ['Chrome', 'Firefox', 'Edge'],
        'MacOS': ['Chrome', 'Firefox', 'Safari'],
        'iOS': ['Safari', 'Chrome'],
        'Android': ['Chrome', 'Firefox']
    }

    os_resolution_compatibility = {
        'Windows': [
            '1920x1080',  # Full HD
            '1366x768',   # HD
            '2560x1440',  # 2K
            '3840x2160',  # 4K
            '1440x900',   # Common laptop
            '1536x864',   # HD+
            '2560x1600',  # MacBook Pro
            '1680x1050'   # HD+
        ],
        'MacOS': [
            '1920x1080',  # Full HD
            '2560x1440',  # 2K
            '2880x1800',  # MacBook Pro
            '2560x1600',  # MacBook Pro
            '3456x2234',  # MacBook Pro 16"
            '3024x1964',  # MacBook Pro 14"
            '2560x1664'   # MacBook Air
        ],
        'iOS': [
            '2778x1284',  # iPhone 13 Pro Max
            '2532x1170',  # iPhone 13, 13 Pro
            '2340x1080',  # iPhone 13 mini
            '2688x1242',  # iPhone 11 Pro Max
            '1792x828',   # iPhone 11
            '2436x1125',  # iPhone 11 Pro
            '2048x2732',  # iPad Pro 12.9"
            '1668x2388',  # iPad Pro 11"
            '1640x2360',  # iPad Air
            '1620x2160',  # iPad 10.9"
            '1536x2048',  # iPad Mini
            '375x812',    # iPhone X/XS/11 Pro (scaled)
            '414x896',    # iPhone XR/XS Max (scaled)
            '390x844',    # iPhone 12/13 (scaled)
            '428x926'     # iPhone 12/13 Pro Max (scaled)
        ],
        'Android': [
            '2400x1080',  # Samsung S21
            '3200x1440',  # Samsung S21 Ultra
            '2340x1080',  # Pixel 6
            '3120x1440',  # Pixel 6 Pro
            '2208x1768',  # Samsung Fold
            '2460x1080',  # OnePlus 9
            '3168x1440',  # OnePlus 9 Pro
            '2400x1080',  # Xiaomi Mi 11
            '3200x1440',  # Xiaomi Mi 11 Ultra
            '2560x1600',  # Samsung Tab S7
            '2800x1752',  # Samsung Tab S7+
            '2000x1200',  # Generic Android Tablet
            '375x812',    # Common Android (scaled)
            '412x915',    # Common Android (scaled)
            '360x800'     # Common Android (scaled)
        ]
    }

    # IP ranges by region
    ip_regions = {
        'North America': {
            'ranges': [
                ('24.0.0.0', '24.255.255.255'),  # Comcast
                ('64.0.0.0', '64.255.255.255'),  # AT&T
                ('98.0.0.0', '98.255.255.255'),  # Verizon
                ('174.0.0.0', '174.255.255.255')  # Charter
            ],
            'countries': ['US', 'CA'],
            'languages': ['en-US', 'en-CA', 'fr-CA']
        },
        'Europe': {
            'ranges': [
                ('81.0.0.0', '81.255.255.255'),  # Deutsche Telekom
                ('88.0.0.0', '88.255.255.255'),  # Orange
                ('92.0.0.0', '92.255.255.255'),  # BT
                ('178.0.0.0', '178.255.255.255')  # Vodafone
            ],
            'countries': ['UK', 'DE', 'FR', 'IT', 'ES'],
            'languages': ['en-GB', 'de-DE', 'fr-FR', 'it-IT', 'es-ES']
        },
        'Asia Pacific': {
            'ranges': [
                ('103.0.0.0', '103.255.255.255'),  # China Mobile
                ('116.0.0.0', '116.255.255.255'),  # NTT
                ('122.0.0.0', '122.255.255.255'),  # SoftBank
                ('180.0.0.0', '180.255.255.255')  # Telstra
            ],
            'countries': ['JP', 'CN', 'KR', 'AU', 'IN'],
            'languages': ['ja-JP', 'zh-CN', 'ko-KR', 'en-AU', 'hi-IN']
        }
    }

    def generate_ip_and_region():
        # Select random region
        region = np.random.choice(list(ip_regions.keys()))
        # Select random IP range from region
        ranges = ip_regions[region]['ranges']
        ip_range = ranges[np.random.randint(0, len(ranges))]  # Fixed: randomly select from ranges list

        # Generate IP within range
        start_ip = int(''.join([f"{int(x):08b}" for x in ip_range[0].split('.')]), 2)
        end_ip = int(''.join([f"{int(x):08b}" for x in ip_range[1].split('.')]), 2)
        ip_int = np.random.randint(start_ip, end_ip)
        ip = '.'.join([str((ip_int >> (8 * i)) & 255) for i in range(3, -1, -1)])

        # Select random country and language from region
        country = np.random.choice(ip_regions[region]['countries'])
        language = np.random.choice(ip_regions[region]['languages'])
        return ip, region, country, language

    data = {
        'session_id': range(1, n_samples + 1),
        'user_id': np.random.randint(1000, 9999, n_samples),
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='min'),
        'ip_address': [],
        'region': [],
        'country': [],
        'language': [],
        'os': np.random.choice(os_types, n_samples),
        'browser': [],
        'screen_resolution': [],
        'user_agent_string': []
    }

    # Generate IP addresses and related data
    for i in range(n_samples):
        ip, region, country, language = generate_ip_and_region()
        data['ip_address'].append(ip)
        data['region'].append(region)
        data['country'].append(country)
        data['language'].append(language)

        os_type = data['os'][i]
        has_inconsistency = np.random.choice([0, 1], p=[0.95, 0.05], size=n_samples)

        if has_inconsistency[i]:
            browser = np.random.choice(browsers)
            resolution = np.random.choice(screen_resolutions)
        else:
            browser = np.random.choice(os_browser_compatibility[os_type])
            resolution = np.random.choice(os_resolution_compatibility[os_type])

        data['browser'].append(browser)
        data['screen_resolution'].append(resolution)
        data['user_agent_string'].append(f"Mozilla/5.0 ({os_type}) AppleWebKit/537.36 {browser}")

    df = pd.DataFrame(data)

    df['is_suspicious'] = 0

    # Rule 1: Browser not compatible with OS
    for os_type, compatible_browsers in os_browser_compatibility.items():
        df.loc[(df['os'] == os_type) & (~df['browser'].isin(compatible_browsers)), 'is_suspicious'] = 1

    # Rule 2: Screen resolution not compatible with OS
    for os_type, compatible_resolutions in os_resolution_compatibility.items():
        df.loc[(df['os'] == os_type) & (~df['screen_resolution'].isin(compatible_resolutions)), 'is_suspicious'] = 1

    # Rule 3: User agent string doesn't match OS or browser
    for idx, row in df.iterrows():
        if not (row['os'] in row['user_agent_string'] and row['browser'] in row['user_agent_string']):
            df.loc[idx, 'is_suspicious'] = 1

    # Rule 4: Language doesn't match region
    for region, data in ip_regions.items():
        df.loc[(df['region'] == region) &
               (~df['language'].isin(data['languages'])), 'is_suspicious'] = 1

    # Rule 5: Unusual region-browser combinations
    df.loc[(df['region'] == 'Asia Pacific') &
           (df['browser'] == 'Edge'), 'is_suspicious'] = 1

    return df
