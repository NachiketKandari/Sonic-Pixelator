import sys
import os
import argparse
import numpy as np
from PIL import Image
import mimetypes
import struct
import math

class SonicPixelator:
    MAGIC_SNIC = bytes([0x53, 0x4E, 0x49, 0x43]) # "SNIC"
    MAGIC_SNIZ = bytes([0x53, 0x4E, 0x49, 0x5A]) # "SNIZ"
    MAGIC_SNIH = bytes([0x53, 0x4E, 0x49, 0x48]) # "SNIH"
    RESERVED_HEADER_PIXELS = 32

    def __init__(self):
        pass

    def _imul(self, a, b):
        # 32-bit signed integer multiplication
        return ((a * b) & 0xFFFFFFFF)
        # However, JS Math.imul(a, b) handles overflows differently.
        # Let's use a more accurate version for signed 32-bit:
        # res = (a * b) & 0xFFFFFFFF
        # return res if res <= 0x7FFFFFFF else res - 0x100000000

    def _get_random_generator(self, seed_val):
        # Seed initialized as: 1337 ^ 0xDEADBEEF
        # Replicating the JS logic exactly:
        # let t = seed += 0x6D2B79F5;
        # t = Math.imul(t ^ (t >>> 15), t | 1);
        # t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        # return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        
        state = [seed_val]
        
        def random():
            # JS >>> 0 makes it unsigned 32-bit.
            # JS ^, +, >>> are 32-bit operators.
            
            # seed += 0x6D2B79F5
            state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
            t = state[0]
            
            # t = Math.imul(t ^ (t >>> 15), t | 1)
            term1 = (t ^ (t >> 15)) & 0xFFFFFFFF
            term2 = (t | 1) & 0xFFFFFFFF
            t = (term1 * term2) & 0xFFFFFFFF
            
            # t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
            term3 = (t ^ (t >> 7)) & 0xFFFFFFFF
            term4 = 61
            imul_res = (term3 * term4) & 0xFFFFFFFF
            t = (t ^ ((t + imul_res) & 0xFFFFFFFF)) & 0xFFFFFFFF
            
            # ((t ^ (t >>> 14)) >>> 0) / 4294967296
            res = (t ^ (t >> 14)) & 0xFFFFFFFF
            return res / 4294967296.0
            
        return random

    def _get_shuffled_indices(self, total_pixels, reserved_pixels):
        count = total_pixels - reserved_pixels
        indices = np.arange(reserved_pixels, total_pixels, dtype=np.uint32)
        
        seed_val = (1337 ^ 0xDEADBEEF) & 0xFFFFFFFF
        rng = self._get_random_generator(seed_val)
        
        for i in range(count - 1, 0, -1):
            j = math.floor(rng() * (i + 1))
            indices[i], indices[j] = indices[j], indices[i]
        
        return indices

    def _apply_opap(self, original, modified, bpc):
        delta = modified - original
        interval = 1 << bpc
        limit = 1 << (bpc - 1)
        
        if delta > limit and (modified - interval) >= 0:
            return modified - interval
        elif delta < -limit and (modified + interval) <= 255:
            return modified + interval
        return modified

    def create_payload(self, audio_data, file_name, mime_type):
        magic = self.MAGIC_SNIC
        data_len = len(audio_data)
        
        mime_bytes = mime_type.encode('utf-8')
        name_bytes = file_name.encode('utf-8')
        
        # Header: magic (4), data_len (4), mime_len (1), mime, name_len (1), name
        header_size = 4 + 4 + 1 + len(mime_bytes) + 1 + len(name_bytes)
        payload = bytearray(header_size + data_len)
        
        offset = 0
        payload[offset:offset+4] = magic
        offset += 4
        
        struct.pack_into('<I', payload, offset, data_len)
        offset += 4
        
        payload[offset] = len(mime_bytes)
        offset += 1
        payload[offset:offset+len(mime_bytes)] = mime_bytes
        offset += len(mime_bytes)
        
        payload[offset] = len(name_bytes)
        offset += 1
        payload[offset:offset+len(name_bytes)] = name_bytes
        offset += len(name_bytes)
        
        payload[offset:offset+data_len] = audio_data
        return payload

    def encode_to_noise(self, payload):
        total_pixels = math.ceil(len(payload) / 3)
        width = math.ceil(math.sqrt(total_pixels))
        height = math.ceil(total_pixels / width)
        
        image_data = np.zeros((height * width, 4), dtype=np.uint8)
        image_data[:, 3] = 255 # Alpha
        
        # Flatten payload into RGB
        padded_payload = bytearray(payload)
        padding_needed = (height * width * 3) - len(payload)
        if padding_needed > 0:
            padded_payload.extend([0] * padding_needed)
            
        payload_np = np.frombuffer(padded_payload, dtype=np.uint8).reshape(-1, 3)
        image_data[:len(payload_np), :3] = payload_np
        
        img = Image.fromarray(image_data.reshape((height, width, 4)), 'RGBA')
        return img

    def encode_to_stego(self, payload, cover_image_path, target_bpc=3):
        with Image.open(cover_image_path) as img:
            img = img.convert('RGB')
            width, height = img.size
            
            total_payload_bits = len(payload) * 8
            available_pixels = (width * height) - self.RESERVED_HEADER_PIXELS
            
            min_required_bpc = math.ceil(total_payload_bits / (available_pixels * 3))
            
            scale = 1.0
            if min_required_bpc > target_bpc:
                desired_channels = math.ceil(total_payload_bits / target_bpc)
                desired_pixels = math.ceil(desired_channels / 3) + self.RESERVED_HEADER_PIXELS
                scale = math.sqrt(desired_pixels / (width * height))
            
            utilization = total_payload_bits / (available_pixels * 3 * target_bpc)
            if utilization > 0.5 and scale == 1.0:
                scale = 1.25
                
            scale = max(1.0, scale)
            new_width = math.ceil(width * scale)
            new_height = math.ceil(height * scale)
            
            if scale > 1.0:
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            pixels = np.array(img).astype(np.int32)
            # Add Alpha channel
            pixels_rgba = np.full((new_height, new_width, 4), 255, dtype=np.uint8)
            pixels_rgba[:, :, :3] = pixels.astype(np.uint8)
            pixels = pixels_rgba.reshape(-1, 4)

            # --- Write Header ---
            header_bits = []
            for b in self.MAGIC_SNIH:
                for bit in range(8):
                    header_bits.append((b >> bit) & 1)
            
            data_len = len(payload)
            for bit in range(32):
                header_bits.append((data_len >> bit) & 1)
            
            for bit in range(8):
                header_bits.append((target_bpc >> bit) & 1)
                
            channel_idx = 0
            pixel_idx = 0
            for bit in header_bits:
                p_idx = pixel_idx
                pixels[p_idx, channel_idx] = (pixels[p_idx, channel_idx] & ~1) | bit
                channel_idx += 1
                if channel_idx > 2:
                    channel_idx = 0
                    pixel_idx += 1

            # --- Write Payload ---
            shuffled_indices = self._get_shuffled_indices(new_width * new_height, self.RESERVED_HEADER_PIXELS)
            
            payload_byte_idx = 0
            payload_bit_idx = 0
            shuffle_arr_idx = 0
            mask = (1 << target_bpc) - 1
            
            while payload_byte_idx < len(payload):
                if shuffle_arr_idx >= len(shuffled_indices):
                    break
                
                target_pixel_idx = shuffled_indices[shuffle_arr_idx]
                
                for c in range(3):
                    bits_to_embed = 0
                    for b in range(target_bpc):
                        if payload_byte_idx < len(payload):
                            bit = (payload[payload_byte_idx] >> payload_bit_idx) & 1
                            bits_to_embed |= (bit << b)
                            payload_bit_idx += 1
                            if payload_bit_idx == 8:
                                payload_bit_idx = 0
                                payload_byte_idx += 1
                    
                    original_val = pixels[target_pixel_idx, c]
                    modified_val = (original_val & ~mask) | bits_to_embed
                    if target_bpc < 8:
                        modified_val = self._apply_opap(original_val, modified_val, target_bpc)
                    
                    pixels[target_pixel_idx, c] = modified_val
                    if payload_byte_idx >= len(payload):
                        break
                shuffle_arr_idx += 1

            return Image.fromarray(pixels.reshape((new_height, new_width, 4)), 'RGBA')

    def parse_payload(self, buffer):
        if len(buffer) < 4:
            raise ValueError("Buffer too small")
            
        magic = buffer[:4]
        is_snic = magic == self.MAGIC_SNIC
        is_sniz = magic == self.MAGIC_SNIZ
        
        if not is_snic and not is_sniz:
            raise ValueError("Invalid Magic Number")
            
        offset = 4
        data_size = struct.unpack_from('<I', buffer, offset)[0]
        offset += 4
        
        mime_len = buffer[offset]
        offset += 1
        mime_type = buffer[offset:offset+mime_len].decode('utf-8')
        offset += mime_len
        
        name_len = buffer[offset]
        offset += 1
        file_name = buffer[offset:offset+name_len].decode('utf-8')
        offset += name_len
        
        data = buffer[offset:offset+data_size]
        return data, file_name, mime_type

    def decode(self, image_path):
        with Image.open(image_path) as img:
            img = img.convert('RGBA')
            pixels = np.array(img).reshape(-1, 4)
            width, height = img.size
            
            # Check for Standard Raw
            raw_check = bytes([pixels[0, 0], pixels[0, 1], pixels[0, 2], pixels[1, 0]])
            if raw_check == self.MAGIC_SNIC or raw_check == self.MAGIC_SNIZ:
                extracted = bytearray(width * height * 3)
                extracted_ptr = 0
                for i in range(width * height):
                    extracted[extracted_ptr] = pixels[i, 0]
                    extracted[extracted_ptr+1] = pixels[i, 1]
                    extracted[extracted_ptr+2] = pixels[i, 2]
                    extracted_ptr += 3
                return self.parse_payload(extracted)
            
            # Check for Stego
            stego_magic = bytearray(4)
            channel_idx = 0
            pixel_idx = 0
            for byte_i in range(4):
                val = 0
                for bit_i in range(8):
                    p_idx = pixel_idx
                    bit = pixels[p_idx, channel_idx] & 1
                    val |= (bit << bit_i)
                    channel_idx += 1
                    if channel_idx > 2:
                        channel_idx = 0
                        pixel_idx += 1
                stego_magic[byte_i] = val
            
            if stego_magic == self.MAGIC_SNIH:
                # Read Len
                len_bytes = bytearray(4)
                for byte_i in range(4):
                    val = 0
                    for bit_i in range(8):
                        p_idx = pixel_idx
                        bit = pixels[p_idx, channel_idx] & 1
                        val |= (bit << bit_i)
                        channel_idx += 1
                        if channel_idx > 2:
                            channel_idx = 0
                            pixel_idx += 1
                    len_bytes[byte_i] = val
                payload_len = struct.unpack('<I', len_bytes)[0]
                
                # Read BPC
                bpc = 0
                for bit_i in range(8):
                    p_idx = pixel_idx
                    bit = pixels[p_idx, channel_idx] & 1
                    bpc |= (bit << bit_i)
                    channel_idx += 1
                    if channel_idx > 2:
                        channel_idx = 0
                        pixel_idx += 1
                
                shuffled_indices = self._get_shuffled_indices(width * height, self.RESERVED_HEADER_PIXELS)
                
                payload = bytearray(payload_len)
                payload_byte_idx = 0
                payload_bit_idx = 0
                current_byte = 0
                shuffle_arr_idx = 0
                mask = (1 << bpc) - 1
                
                while payload_byte_idx < payload_len:
                    if shuffle_arr_idx >= len(shuffled_indices):
                        break
                    target_pixel_idx = shuffled_indices[shuffle_arr_idx]
                    for c in range(3):
                        bits = pixels[target_pixel_idx, c] & mask
                        for b in range(bpc):
                            bit = (bits >> b) & 1
                            current_byte |= (bit << payload_bit_idx)
                            payload_bit_idx += 1
                            if payload_bit_idx == 8:
                                payload[payload_byte_idx] = current_byte
                                payload_byte_idx += 1
                                payload_bit_idx = 0
                                current_byte = 0
                                if payload_byte_idx >= payload_len: break
                        if payload_byte_idx >= payload_len: break
                    shuffle_arr_idx += 1
                
                return self.parse_payload(payload)
                
            raise ValueError("No Sonic Pixelator data found")

def main():
    parser = argparse.ArgumentParser(description="Sonic Pixelator - Audio-to-Image Steganography (Python Port)")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Encode
    enc_parser = subparsers.add_parser("encode", help="Encode audio into image")
    enc_parser.add_argument("audio", help="Path to input audio file")
    enc_parser.add_argument("--cover", help="Path to cover image (optional, enables Stego mode)")
    enc_parser.add_argument("--bpc", type=int, default=3, help="Bits per channel for Stego (1-7)")
    enc_parser.add_argument("--out", required=True, help="Output image path (PNG)")
    
    # Decode
    dec_parser = subparsers.add_parser("decode", help="Decode audio from image")
    dec_parser.add_argument("image", help="Path to Sonic Pixelator image")
    dec_parser.add_argument("--out", help="Output audio path (optional, uses filename from header if omitted)")
    
    args = parser.parse_args()
    
    sp = SonicPixelator()
    
    if args.command == "encode":
        with open(args.audio, 'rb') as f:
            audio_data = f.read()
            
        mime_type, _ = mimetypes.guess_type(args.audio)
        if not mime_type:
            mime_type = "audio/mpeg"
        
        file_name = os.path.basename(args.audio)
        payload = sp.create_payload(audio_data, file_name, mime_type)
        
        if args.cover:
            img = sp.encode_to_stego(payload, args.cover, args.bpc)
        else:
            img = sp.encode_to_noise(payload)
            
        img.save(args.out, "PNG")
        print(f"Encoded into {args.out}")
        
    elif args.command == "decode":
        try:
            audio_data, file_name, mime_type = sp.decode(args.image)
            out_path = args.out if args.out else file_name
            with open(out_path, 'wb') as f:
                f.write(audio_data)
            print(f"Decoded into {out_path} (Type: {mime_type})")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
