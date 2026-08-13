from ragstream.utils import check_services

if not check_services():
    raise RuntimeError("Required backend services (GROBID and llama.cpp) are not running.")