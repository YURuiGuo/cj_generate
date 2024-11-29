from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration
from accelerate import Accelerator
import torch
import os

# accelerate launch 
def process_candidates_and_generate_files(candidate_file,checkpoint, save_path, num_lines_to_include=3):
    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)

    accelerator = Accelerator()
    model = accelerator.prepare(model)

    model.eval()

    # distribute data
    process_index = accelerator.process_index
    num_processes = accelerator.num_processes

    files_per_process = len(candidate_file) // num_processes
    start_idx = process_index * files_per_process
    end_idx = start_idx + files_per_process

    if process_index == num_processes - 1:
        end_idx = len(candidate_file)# 最后一个进程处理剩余的文件


    for i in range(start_idx, end_idx):

        candidate = candidate_file[i]

        updated_code_lines = infer_and_insert(candidate, model, tokenizer, accelerator)

        updated_code = '\n'.join(updated_code_lines)

        generated_file_path = os.path.join(save_path, f"seed_file_{i + 1}.txt")
        with open(generated_file_path, 'w') as f:
            f.write(updated_code)

        print(f"Generated file {i + 1}: {generated_file_path}")


def infer_and_insert(candidate, model, tokenizer, accelerator):

    code_lines = candidate["code"].split('\n')
    insert_points = candidate["points"]

    updated_code_lines = code_lines.copy()  # 保持原代码不变，需要Index定位插入点

    for point in insert_points:
        if 0 <= point < len(code_lines):

            end_point = min(point + 3, len(code_lines))  # 默认插入点附近的3行作为prompt
            input_code = '\n'.join(code_lines[point:end_point])

            encoded_input = tokenizer.encode(input_code + "<extra_id_0>", return_tensors="pt")
            encoded_input = encoded_input.to(accelerator.device) 
            try:
                with torch.no_grad():
                    outputs = model.module.generate(encoded_input, max_length=100)

            except RuntimeError as e:
                    print('* OOM: {}, {}'.format(candidate["code"], input_code))
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

           
            output_lines = generated_code.split('\n') # 替换 √ / 插入 × / 变异 ×
            for i, output_line in enumerate(output_lines):
                if end_point + i < len(updated_code_lines):
                    updated_code_lines[end_point + i] = output_line
                else:
                    updated_code_lines.append(output_line)

    return updated_code_lines







def test():
    # 一个元素处理

    corpus_path = r"/data1/amax/CodeT5_FT/projectByCo/Corpus"
    save_path = r"/data1/amax/CodeT5_FT/Generate_file/files"
    checkpoint = r"/data1/amax/CodeT5_FT/CodeT5_2b/output_dir"

    can = [
    {
        "code": "import std.database.sql.*\nimport std.io.*\nimport std.time.*\nimport std.regex.*\nimport odbc4cj.*\nimport std.unittest.*\nimport std.unittest.testmacro.*\nmain(): Int64 {\n    let filterResultTest: OdbcTest = OdbcTest()\n    filterResultTest.test001()\n    return 0\n}\n        @Assert(\"SqlNullableReal\", queryResult.columnInfos[1].typeName)\n        match (arr[1]) {\n            case v: SqlNullableReal => @Assert(None, v.value)\n            case _ => @Assert(\"abc\", \"123\")\n        }\n        queryResult.close()\n    }\n}\n// /data1/amax/CodeT5_FT/projectByCo/Corpus/odbc4cj-develop/test/LLT/real_test_001.cj, Insert points:[10, 11]",
        "points": [
            10,
            11
        ]
    },
    {
        "code": "import yaml4cj.yaml.*\nimport std.collection.*\nmain() {\n    return 0\n}\n// /data1/amax/CodeT5_FT/projectByCo/Corpus/yaml4cj-develop/test/LLT/test_common_token_type.cj, Insert points:[3, 4]",
        "points": [
            3,
            4
        ]
    },
    {
        "code": "from flexSearch4cj import flexSearch4cj.*\nfrom std import unittest.*\nfrom std import unittest.testmacro.*\nfrom std import collection.*\nfrom encoding import json.*\nmain() {\n    var doc = DocumentTestUpdateReadme()\n    doc.execute()\n    doc.printResult()\n    return 0\n}\n        }\n      }\"##)\n    var jvUpdate: JsonValue = JsonValue.fromStr(\n      ##\"{\n        \"tag\":[\"tag4\"],\n        \"str1\":\"val2\",\n        \"arr2\":[\"v223\"]\n      }\"##)\n        @Assert(flag, 0)\n    }\n}\n// /data1/amax/CodeT5_FT/projectByCo/Corpus/flexSearch4cj-develop/flexsearch/src/main/cangjie/test/DOC/test_document_update_readme.cj, Insert points:[7, 19, 20]",
        "points": [
            7,
            19,
            20
        ]
    },
    {
        "code": "import crypto4cj.aescj.*\nimport crypto4cj.utils.*\nimport encoding.base64.*\nimport std.collection.*\nimport std.unicode.*\nmain() {\n    var keys: Array<UInt8> = \"1234567812345678\".toArray()\n    var inside: Array<UInt8> = \"skfhafahglkahglahglkahgalgfssfferere\".toArray()\n    var ivec: Array<UInt8> = \"\".toArray()\n    var key = AESKEY()\n    var outside: Array<UInt8> = Array<UInt8>(inside.size, repeat: 0)\n    aesSetEncryptKey(keys, 128, key)\n    try {\n        aesCfb1Encrypt(inside, outside, key, ivec, AES_ENCRYPT)\n    }catch(e: Exception) {\n        println(e.toString())\n        return 0\n    }\n    return 1\n}\n// /data1/amax/CodeT5_FT/projectByCo/Corpus/crypto4cj-develop/test/LLT/aes/aes_cfb1_04_test.cj, Insert points:[8, 12, 15]",
        "points": [
            8,
            12,
            15
        ]
    },
    {
        "code": "import crypto4cj.aescj.*\nimport crypto4cj.utils.*\nimport encoding.base64.*\nimport std.collection.*\nimport std.unicode.*\nmain() {\n    var keys: Array<UInt8> = \"1234567812345678\".toArray()\n    var inside: Array<UInt8> = \"skfhafahglkahglahglkahgalgfssfferere\".toArray()\n    var ivec: Array<UInt8> = \"0000000000000000\".toArray()\n    var key = AESKEY()\n    var outside: Array<UInt8> = Array<UInt8>(100, repeat: 0)\n    aesSetEncryptKey(keys, 128, key)\n    try {\n        aesCfb8Encrypt(inside, outside, key, ivec, AES_ENCRYPT)\n    }catch(e: Exception) {\n        println(e.toString())\n        return 0\n    }\n    return 1\n}\n// /data1/amax/CodeT5_FT/projectByCo/Corpus/crypto4cj-develop/test/LLT/aes/aes_cfb8_03_test.cj, Insert points:[16]",
        "points": [
            16
        ]
    },
    {
        "code": "import yaml4cj.yaml.*\nimport std.collection.*\nmain() {\n    return 0\n}\n// /data1/amax/CodeT5_FT/projectByCo/Corpus/yaml4cj-develop/test/LLT/test_node_type_01.cj, Insert points:[3]",
        "points": [
            3
        ]
    },
    {
        "code": "import std.unittest.*\nimport std.unittest.testmacro.*\nimport matrix4cj.*\nmain(): Int64 {\n    let tester = MatrixTester08()\n    let test = tester.asTestSuite().runTests()\n    test.failedCount + test.errorCount\n}\n}\n// /data1/amax/CodeT5_FT/projectByCo/Corpus/matrix4cj-develop/test/LLT/test_matrix_08.cj, Insert points:[4, 5, 8]",
        "points": [
            4,
            5,
            8
        ]
    },
    {
        "code": "import io4cj.*\nimport std.unittest.*\nimport std.io.*\nimport std.math.*\nimport secodeFuzz.*\nimport std.unittest.testmacro.*\nimport std.collection.ArrayList\nfunc call_fuzz(value: Int64){\n    if (value < Int64.Min || value > Int64.Max) {\n\t\treturn\n\t}\n    var sourceIns:Sink = Buffer()\n    var bufferIns = Okio.buffer(sourceIns)\n    bufferIns.writeInt16Le(value)\n    return\n}\nfunc unboundTest():Unit{\n    var case_name = CString(\"Fuzz_BufferedSink_writeInt16Le\")\n    unsafe{\n        DT_FUZZ_Int64(0,30000000,case_name,0,call_fuzz)\n        case_name.free()\n    }\n}\nmain(){\n    unboundTest()\n    return\n}\n// /data1/amax/CodeT5_FT/projectByCo/Corpus/io4cj-develop/test/FUZZ/okio/testfuzz_bufferedSink_writeInt16Le.cj, Insert points:[7]",
        "points": [
            7
        ]
    },
    {
        "code": "package hyperloglog_example\nimport std.time.Duration\nimport std.time.DurationExtension\nimport redis_sdk.client.api.*\nimport redis_sdk.client.commands.*\nimport redis_sdk.client.*\nmain() {\n    let redisClient = RedisClientBuilder.builder().host(\"127.0.0.1\").port(6379).password(\"mypassword\").respVersion(3).\n        readTimeout(Duration.second * 60).writeTimeout(Duration.second * 30).receiveBufferSize(32768).sendBufferSize(\n        32768).build()\n    let key1 = \"redisExampleTestHyperloglogKey1\"\n    let value1 = \"redisExampleTestHyperloglogValue1\"\n    let value2 = \"redisExampleTestHyperloglogValue2\"\n    let value3 = \"redisExampleTestHyperloglogValue3\"\n    println(\"PFADD ${key1} ${value1} ${value2} ${value3}\")\n    var res = redisClient.pfadd(key1, value1, value2, value3)\n    println(res)\n    println(\"PFCOUNT ${key1}\")\n    var res1 = redisClient.pfcount(key1)\n    println(res1)\n}\n// /data1/amax/CodeT5_FT/projectByCo/Corpus/redis-sdk-master/samples/hyperloglog_example/src/hyperloglog_example.cj, Insert points:[7, 14, 18]",
        "points": [
            7,
            14,
            18
        ]
    },
    {
        "code": "import crypto4cj.aescj.*\nimport crypto4cj.utils.*\nimport encoding.base64.*\nimport std.collection.*\nimport std.unicode.*\nmain() {\n    var keys: Array<UInt8> = \"1234567812345678\".toArray()\n    var inside: Array<UInt8> = \"\".toArray()\n    var ivec: Array<UInt8> = \"0000000000000000\".toArray()\n    var key = AESKEY()\n    var outside: Array<UInt8> = Array<UInt8>(Int64(AES_BLOCK_SIZE), repeat: 0)\n    aesSetEncryptKey(keys, 128, key)\n    try {\n        aesCbcEncrypt(inside, outside, key, ivec, AES_ENCRYPT)\n    }catch(e: Exception) {\n        println(e.toString())\n        return 0\n    }\n    return -1\n}\n// /data1/amax/CodeT5_FT/projectByCo/Corpus/crypto4cj-develop/test/LLT/aes/aes_cbc_07_test.cj, Insert points:[11, 13]",
        "points": [
            11,
            13
        ]
    },
    {
        "code": "import mysqlclient_ffi.*\nimport std.unittest.*\nimport std.unittest.testmacro.*\nimport std.database.sql.*\nimport std.time.*\nmain(): Int64 {\n    let mysqlDoubleTest: MysqlDoubleTest = MysqlDoubleTest()\n    mysqlDoubleTest.mysqlDoubleTest01()\n    return 0\n}\n        id = SqlBigInt(3)\n        data1 = SqlDouble(0.00)\n        data2 = SqlNullableDouble(0.00)\n        arrDb = [id, data1, data2]\n        isBool = mysqlQueryResult.next(arrDb)\n        @Assert(false, isBool)\n        mysqlStatement4.close()\n        let mysqlStatement5: MysqlStatement = mysqlConnection.prepareStatement(\"delete from t_test_double where id = ?\")\n        @Assert(1, mysqlStatement5.parameterCount)\n        id = SqlBigInt(1)\n        arrDb = [id]\n        let mysqlUpdateResult4: MysqlUpdateResult = mysqlStatement5.update(arrDb)\n        @Assert(1, mysqlUpdateResult4.rowCount)\n        mysqlStatement5.close()\n        let mysqlStatement6: MysqlStatement = mysqlConnection.prepareStatement(\n            \"select * from t_test_double where id = 1\")\n        let mysqlQueryResult1: MysqlQueryResult = mysqlStatement6.query()\n        id = SqlBigInt(3)\n        data1 = SqlDouble(0.00)\n        data2 = SqlNullableDouble(0.00)\n        arrDb = [id, data1, data2]\n        isBool = mysqlQueryResult1.next(arrDb)\n        @Assert(false, isBool)\n        mysqlStatement6.close()\n        let mysqlStatement7: MysqlStatement = mysqlConnection.prepareStatement(\n            \"update t_test_double set value1 = ?, value2 = ?  where id = ?\")\n        @Assert(3, mysqlStatement7.parameterCount)\n        id = SqlBigInt(2)\n        data1 = SqlDouble(2.222)\n        data2 = SqlNullableDouble(4.444)\n        arrDb = [data1, data2, id]\n        let mysqlUpdateResult5: MysqlUpdateResult = mysqlStatement7.update(arrDb)\n        @Assert(1, mysqlUpdateResult5.rowCount)\n        mysqlStatement7.close()\n        let mysqlStatement8: MysqlStatement = mysqlConnection.prepareStatement(\n            \"select * from t_test_double where id = 2\")\n        var mysqlQueryResult2: MysqlQueryResult = mysqlStatement8.query()\n        id = SqlBigInt(3)\n        data1 = SqlDouble(0.00)\n        data2 = SqlNullableDouble(0.00)\n        arrDb = [id, data1, data2]\n        isBool = mysqlQueryResult2.next(arrDb)\n        @Assert(true, isBool)\n        @Assert(2, (arrDb[0] as SqlBigInt).getOrThrow().value)\n        @Assert(2.222, (arrDb[1] as SqlDouble).getOrThrow().value)\n        @Assert(4.444, (arrDb[2] as SqlNullableDouble).getOrThrow().value)\n        mysqlStatement8.close()\n        let mysqlStatement9: MysqlStatement = mysqlConnection.prepareStatement(\"drop table if exists t_test_double\")\n        mysqlStatement9.update()\n        mysqlStatement9.close()\n        mysqlConnection.close()\n    }\n}\n// /data1/amax/CodeT5_FT/projectByCo/Corpus/mysqlclient-ffi-develop/test/LLT/mysql_basic_double_null_test.cj, Insert points:[31, 59, 60]",
        "points": [
            31,
            59,
            60
        ]
    },
    {
        "code": "import uuid4cj.uuid4cj.*\nimport std.unittest.*\nimport std.unittest.testmacro.*\nimport std.random.*\nmain() {\n    let testReadmeExample = TestReadmeExample()\n    testReadmeExample.test1()\n    return 0\n}\n}\n// /data1/amax/CodeT5_FT/projectByCo/Corpus/uuid4cj-develop/test/DOC/readme_example_timeEpoch2.cj, Insert points:[6, 7, 8]",
        "points": [
            6,
            7,
            8
        ]
    }]
    process_candidates_and_generate_files(can,checkpoint, save_path)



if __name__ == "__main__":
    test()
