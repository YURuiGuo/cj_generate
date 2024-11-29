# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# import torch

# # 16B 是最佳的模型，但2B适合当前显存
# checkpoint = "/data1/amax/CodeT5_FT/CodeT5_2b/model_770"
# device = "cpu" # for GPU usage or "cpu" for CPU usage

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
#                                               torch_dtype=torch.float32,
#                                               low_cpu_mem_usage=True,
#                                               trust_remote_code=True).to(device)

# encoding = tokenizer("def print_hello_world():", return_tensors="pt").to(device)
# encoding['decoder_input_ids'] = encoding['input_ids'].clone()
# outputs = model.generate(**encoding, max_length=30)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))



from transformers import T5ForConditionalGeneration, AutoTokenizer

checkpoint = "/data1/amax/CodeT5_FT/CodeT5_2b/output_dir"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

# inputs = tokenizer.encode("    func clientFirstMessage(): Array<Byte> {\n        let clientNonceStr = \"n=,r=${String.fromUtf8(clientNonce)}<extra_id_0>", return_tensors="pt").to(device)

inputs = tokenizer.encode("    public func blpop(timeout: Int64, keys: Array<String>): ParameterizedRedisCommand<?(String, String)> {\n        let commandArgs = CommandArgs().key(keys)\n        commandArgs.add(timeout)\n        return ParameterizedRedisCommand<?(String, String)>(<extra_id_0>", return_tensors="pt").to(device)

# inputs = tokenizer.encode("""
# @Test
# public void test0()  throws Throwable  {
#       MyTestableClass myTestableClass0 = new MyTestableClass();
#       String string0 = myTestableClass0.compress("<extra_id_0>
# """, return_tensors="pt").to(device)

#(catlm) amax@amax-Super-Server:/data1/amax/CodeT5_FT$ python3 CodeT5_2b/code/interfer.py 


outputs = model.generate(inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# ==> print "Hello World"

"""

        member.ForEach(memberString => commandArgs.sadd(memberString))
        return commandArgs.execute();
    }

    public func srem(key: String, member: Array<String>): ParameterizedRedisCommand<Int64> {
        let commandArgs = CommandArgs().key(key)
        member.ForEach(memberString => commandArgs.srem(memberString))
        return commandArgs.execute();
    }

    public func sremrange(key: String, start: Int64, end: Int64): ParameterizedRedisCommand<Int64> {
        let commandArgs = CommandArgs().key(key)
        start.ForEach(startInt64 => commandArgs.sremrange(startInt64, endInt64))
        return commandArgs.execute();
    }

    public func sremrangebyscore(key: String, start: Int64, end: Int64): ParameterizedRedisCommand<Int64> {
        let commandArgs = CommandArgs().key(key)
        start.ForEach(startInt64 => commandArgs.sremrangebyscore(startInt64, endInt64))
        return commandArgs.execute();
    }

    public func sremrangebyscoreandscore(key: String, start: Int64, end: Int64): ParameterizedRedisCommand<Int64> {
        let commandArgs = CommandArgs().key(key)
        start.ForEach(startInt64 => commandArgs.sremrangebyscoreandscore(startInt64, endInt64))
        return commandArgs.execute();
    }

    public func sremrangebyscoreandscorerange(key: String, start: Int64, end: Int64): ParameterizedRedisCommand<Int64> {
        let commandArgs = CommandArgs().key(key)
        start.ForEach(startInt64 => commandArgs.sremrangebyscoreandscorerange(startInt64, endInt64))
        return commandArgs.execute();
    }

    public func sremrangebyscorerange(key: String, start: Int64, end: Int64): ParameterizedRedisCommand<Int64> {
        let commandArgs = CommandArgs().key(key)
        start.For
"""

