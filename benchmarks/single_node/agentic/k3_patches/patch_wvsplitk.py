F="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py"
s=open(F).read(); orig=s
# guard 1: wvSplitKrc branch
a='''    if use_skinny_reduce_counting:
        return ops.wvSplitKrc(x, weight, cu_count, bias)'''
b='''    if use_skinny_reduce_counting:
        # PATCH(vLLM #50618): skinny GEMMs index operands linearly and do not
        # accept strides -> OOB read on strided activations (the cudagraph
        # capture fault). Force contiguous.
        if not x.is_contiguous():
            x = x.contiguous()
        return ops.wvSplitKrc(x, weight, cu_count, bias)'''
assert a in s, "wvSplitKrc anchor"; s=s.replace(a,b,1)
# guard 2: wvSplitK branch
c='''    if use_skinny:
        x_view = x.reshape(-1, x.size(-1))'''
d='''    if use_skinny:
        x_view = x.reshape(-1, x.size(-1))
        # PATCH(vLLM #50618): wvSplitK indexes linearly; strided activation -> OOB.
        if not x_view.is_contiguous():
            x_view = x_view.contiguous()'''
assert c in s, "wvSplitK anchor"; s=s.replace(c,d,1)
assert s!=orig; open(F,"w").write(s)
print("wvSplitK OOB fix applied; markers:", s.count("PATCH(vLLM #50618)"))
