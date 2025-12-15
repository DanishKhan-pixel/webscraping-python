#!/usr/bin/env python3
"""
Script to convert all print statements to logging statements in scrapper_optimized.py
"""
import re

def convert_prints_to_logging(filepath):
    """Convert print statements to appropriate logging calls"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup original
    with open(filepath + '.backup', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Define replacements (order matters!)
    replacements = [
        # Error messages
        (r'print\(f"❌ ([^"]+)"\)', r'logger.error(f"❌ \1")'),
        (r'print\(f"Failed to load ([^"]+)"\)', r'logger.warning(f"Failed to load \1")'),
        
        # Warning messages
        (r'print\(f"⚠️ ([^"]+)"\)', r'logger.warning(f"⚠️ \1")'),
        (r'print\(f"Skipping ([^"]+)"\)', r'logger.warning(f"Skipping \1")'),
        
        # Info messages - specific patterns
        (r'print\(f"\[Browser\] ([^"]+)"\)', r'logger.info(f"[Browser] \1")'),
        (r'print\(f"\[Link Extraction\] ([^"]+)"\)', r'logger.info(f"[Link Extraction] \1")'),
        (r'print\(f"🔍 ([^"]+)"\)', r'logger.info(f"🔍 \1")'),
        (r'print\(f"✅ ([^"]+)"\)', r'logger.info(f"✅ \1")'),
        (r'print\(f"🔗 ([^"]+)"\)', r'logger.info(f"🔗 \1")'),
        (r'print\(f"🚗 ([^"]+)"\)', r'logger.info(f"🚗 \1")'),
        (r'print\(f"🚀 ([^"]+)"\)', r'logger.info(f"🚀 \1")'),
        (r'print\(f"📊 ([^"]+)"\)', r'logger.info(f"📊 \1")'),
        (r'print\(f"⚡ ([^"]+)"\)', r'logger.info(f"⚡ \1")'),
        (r'print\(f"🎯 ([^"]+)"\)', r'logger.info(f"🎯 \1")'),
        (r'print\(f"📂 ([^"]+)"\)', r'logger.info(f"📂 \1")'),
        (r'print\(f"🌐 ([^"]+)"\)', r'logger.info(f"🌐 \1")'),
        (r'print\(f"📄 ([^"]+)"\)', r'logger.info(f"📄 \1")'),
        (r'print\(f"💾 ([^"]+)"\)', r'logger.info(f"💾 \1")'),
        (r'print\(f"✔ ([^"]+)"\)', r'logger.info(f"✔ \1")'),
        
        # String literals (no f-string)
        (r'print\("VEHICLE SCRAPED"\)', r'logger.info("VEHICLE SCRAPED")'),
        (r'print\("❌ No next page found"\)', r'logger.debug("❌ No next page found")'),
        
        # JSON dumps (should be debug level)
        (r'print\(json\.dumps\(([^)]+)\)\)', r'logger.debug(json.dumps(\1))'),
        
        # Separator lines
        (r'print\(f"\\n\{\'=\'\*60\}"\)', r'logger.info(f"\\n{\'=\'*60}")'),
        (r'print\(f"\{\'=\'\*60\}"\)', r'logger.info(f"{\'=\'*60}")'),
        
        # Any remaining print statements (generic info)
        (r'print\(f"([^"]+)"\)', r'logger.info(f"\1")'),
        (r'print\("([^"]+)"\)', r'logger.info("\1")'),
    ]
    
    # Apply replacements
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Converted all print statements to logging")
    print(f"📁 Backup saved to: {filepath}.backup")

if __name__ == "__main__":
    convert_prints_to_logging("/home/danish/Downloads/webscraping/scrapper_optimized.py")
