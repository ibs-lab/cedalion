import hashlib

file_path = "C:/Users/avonl/OneDrive/Desktop/image_reconstruction_fluence_DOT.pickle.gz"


def compute_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Example usage

print(compute_sha256(file_path))