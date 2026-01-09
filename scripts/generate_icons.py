"""
Generate Chrome extension icons for VeriPulse.
Creates PNG icons at 16x16, 48x48, and 128x128 sizes.
"""

import os

# Simple PNG generator using raw bytes
def create_simple_icon(size, output_path):
    """Create a simple green shield icon as PNG."""
    
    # We'll use PIL if available, otherwise create placeholder
    try:
        from PIL import Image, ImageDraw
        
        # Create image with emerald green background
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw circle background
        padding = size // 8
        draw.ellipse([padding, padding, size - padding, size - padding], 
                     fill=(16, 185, 129))  # Emerald-500
        
        # Draw shield
        shield_padding = size // 4
        center = size // 2
        shield_top = size // 5
        shield_bottom = size - size // 5
        shield_width = size // 3
        
        shield_points = [
            (center, shield_top),  # Top
            (center + shield_width, shield_top + size // 8),  # Top right
            (center + shield_width, center),  # Right
            (center, shield_bottom),  # Bottom
            (center - shield_width, center),  # Left
            (center - shield_width, shield_top + size // 8),  # Top left
        ]
        draw.polygon(shield_points, fill=(255, 255, 255))
        
        # Draw checkmark
        check_start = (center - size // 6, center)
        check_mid = (center - size // 12, center + size // 8)
        check_end = (center + size // 5, center - size // 6)
        
        line_width = max(2, size // 16)
        draw.line([check_start, check_mid], fill=(16, 185, 129), width=line_width)
        draw.line([check_mid, check_end], fill=(16, 185, 129), width=line_width)
        
        img.save(output_path, 'PNG')
        print(f"✅ Created {output_path} ({size}x{size})")
        return True
        
    except ImportError:
        print("⚠️ PIL not available, creating placeholder icons")
        return False


def create_placeholder_icons():
    """Create minimal placeholder icons without PIL."""
    
    # Minimal valid PNG (1x1 green pixel)
    # PNG header + IHDR + IDAT + IEND
    def make_minimal_png(size):
        # This creates a valid but simple PNG
        import struct
        import zlib
        
        def png_chunk(chunk_type, data):
            chunk = chunk_type + data
            return struct.pack('>I', len(data)) + chunk + struct.pack('>I', zlib.crc32(chunk) & 0xffffffff)
        
        # PNG signature
        signature = b'\x89PNG\r\n\x1a\n'
        
        # IHDR chunk
        ihdr_data = struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)  # RGB
        ihdr = png_chunk(b'IHDR', ihdr_data)
        
        # Image data (simple green fill)
        raw_data = b''
        for y in range(size):
            raw_data += b'\x00'  # Filter byte
            for x in range(size):
                raw_data += b'\x10\xb9\x81'  # Emerald green RGB
        
        compressed = zlib.compress(raw_data)
        idat = png_chunk(b'IDAT', compressed)
        
        # IEND chunk
        iend = png_chunk(b'IEND', b'')
        
        return signature + ihdr + idat + iend
    
    return make_minimal_png


def main():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    public_dir = os.path.join(script_dir, '..', 'apps', 'web', 'public')
    
    # Icon sizes
    sizes = [16, 48, 128]
    
    success = True
    for size in sizes:
        output_path = os.path.join(public_dir, f'icon{size}.png')
        if not create_simple_icon(size, output_path):
            success = False
    
    if not success:
        print("\n💡 To create proper icons:")
        print("   1. Install Pillow: pip install Pillow")
        print("   2. Run this script again")
        print("\n   Or manually create icons from icon.svg")


if __name__ == '__main__':
    main()
