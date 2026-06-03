import os
import regex as re
from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge_dicts(dicts: Iterator[dict[bytes, int]]) -> dict[bytes, int]:
    output = defaultdict(int)
    for dic in dicts:
        for key, value in dic.items():
            output[key] += value
    return output


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(chunk: str) -> dict[bytes, int]:
    token_map = defaultdict(int)

    for token_str in re.finditer(PAT, chunk):
        token = token_str.group().encode('utf-8')
        token_map[token] += 1
    return token_map

def process_chunk(data) -> dict[bytes, int]:
    chunk_boundaries, file_path, special_token = data
    special_token_str = special_token.decode('utf-8')
    start, end = chunk_boundaries
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore')
        # split on special tokens - we do not care about their count and we do not want to merge between documents
        subchunks = chunk.split(special_token_str)
        pretokenized_subchunks = [pretokenize_chunk(subchunk) for subchunk in subchunks]
        return merge_dicts(iter(pretokenized_subchunks))

def pretokenize_text(
    file_path: str,
    num_processes: int = 16,
    split_special_token: bytes = b'<|endoftext|>'
) -> dict[bytes, int]:
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

        boundary_pairs = list(zip(boundaries[:-1], boundaries[1:]))
        process_chunk_input_data = list([(pair, file_path, split_special_token) for pair in boundary_pairs])
        with Pool(processes=num_processes) as pool:
            tokenized_chunks: Iterator[dict[bytes, int]] = pool.imap(process_chunk, process_chunk_input_data)
            tokenized_text = merge_dicts(tokenized_chunks)

            return tokenized_text

if __name__ == "__main__":
    map = pretokenize_text(
        file_path='../../data/TinyStoriesV2-GPT4-valid.txt',
        num_processes=1,
    )
    tokens = map.keys()
    len(tokens)