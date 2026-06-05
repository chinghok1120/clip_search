#!/bin/bash
cd ~/clip_search/dbbench
export N=1000000
echo "### N=$N  $(date) ###"
echo "## gen"; ~/clip_search/venv_dbbench/bin/python gen_data.py
echo "## FAISS";   ~/clip_search/venv/bin/python bench_faiss.py
echo "## LANCEDB"; ~/clip_search/venv_dbbench/bin/python bench_lancedb.py 2>/dev/null
echo "## QDRANT";  ~/clip_search/venv_dbbench/bin/python bench_qdrant.py 2>/dev/null
echo "### done $(date) ###"
