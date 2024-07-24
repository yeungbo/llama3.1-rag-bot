# 使用 Llama3.1 实施 RAG

这里我们描述一下使用Llama3.1实现RAG的过程。这包括父/子分块和词汇/语义搜索，用于提高 Advanced RAG 的性能。整体架构如下。

1) 当您使用浏览器访问 CloudFront 的域时，聊天屏幕 UI 是使用 S3 中的 html、css 和 js 构建的。
2) 当用户输入 userId 并连接时，将搜索 DynamoDB 中存储的过去聊天记录并将其显示在屏幕上。
3) 当用户在聊天窗口中输入消息时，该消息将通过支持 WebSocket 的 API 网关传递到 Lambda（聊天）。
4) Lambda（聊天）检查是否存在带有 userId 的聊天历史记录并加载它。
5) 通过组合聊天历史记录和当前问题来创建一个新问题，然后将其嵌入并在矢量存储 OpenSearch 中进行搜索。
6) 我们要求 Llama3 LLM 使用新问题（修订后的问题）和通过 RAG 获得的相关文件作为上下文进行答复。
7) Llama3 生成的答案通过 Lambda（聊天）和 API 网关，并作为 Websocket 传递给客户端。

<img src="./images/basic-architecture.png" width="800">

## Llama3.1 

[Llama3.1 paper](https://scontent-ssn1-1.xx.fbcdn.net/v/t39.2365-6/452387774_1036916434819166_4173978747091533306_n.pdf?_nc_cat=104&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=t6egZJ8QdI4Q7kNvgEBG6o4&_nc_ht=scontent-ssn1-1.xx&oh=00_AYBdfFc8msOH4iSUsYP_7d5LJLfxTrtJ_aV2U5elEF-Ihg&oe=66A60A8D)已经有各种升级，比如Llama3.1论文。. 

![llama3 1](https://github.com/user-attachments/assets/9abf01bf-f044-4bbf-b825-d73035a78287)

Llama 3.1 支持的语言：英语、法语、德语、印地语、意大利语、葡萄牙语、西班牙语、泰语（7 月 24 日）。 Llama3.1(July 24).
Llama3.1 [multilingual](https://scontent-ssn1-1.xx.fbcdn.net/v/t39.2365-6/452387774_1036916434819166_4173978747091533306_n.pdf?_nc_cat=104&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=t6egZJ8QdI4Q7kNvgEBG6o4&_nc_ht=scontent-ssn1-1.xx&oh=00_AYBdfFc8msOH4iSUsYP_7d5LJLfxTrtJ_aV2U5elEF-Ihg&oe=66A60A8D).

## Llama3 RAG 实施


### Llama3.1使用LangChain配置

我们使用LangChain 的 [ChatBedrock](https://python.langchain.com/v0.2/docs/integrations/chat/bedrock/).

```python
boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=bedrock_region,
    config=Config(
        retries = {
            'max_attempts': 30
        }
    )
)
parameters = {
    "max_gen_len": 1024,  
    "top_p": 0.9, 
    "temperature": 0.1
}    

chat = ChatBedrock(   
    model_id=modelId,
    client=boto3_bedrock, 
    model_kwargs=parameters,
)
```

Llama3.1 的型号信息如下。

```java
const llama3 = [
  {
    "bedrock_region": "us-west-2", // Oregon
    "model_type": "llama3",
    "model_id": "meta.llama3-1-70b-instruct-v1:0"
  }
];
```

### 基本聊天

您可以使用提示指定聊天机器人的名称和角色。聊天历史记录使用 MessagesPlaceholder() 反映。

```python
def general_conversation(connectionId, requestId, chat, query):
    if isKorean(query)==True :
        system = (
            "다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다. 답변은 한국어로 합니다."
        )
    else: 
        system = (
            "Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor."
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    
    history = memory_chain.load_memory_variables({})["chat_history"]
                
    chain = prompt | chat    
    try: 
        isTyping(connectionId, requestId)  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)    
                            
        msg = stream.content
        print('msg: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        
    return msg
```

这里，流如下所示提取事件，并将结果以 json 格式传递给客户端。

```python
def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            msg = msg + event

            result = {
                'request_id': requestId,
                'msg': msg,
                'status': 'proceeding'
            }
            sendMessage(connectionId, result)
    return msg
```

### 对话历史管理

当用户连接时，将从 DynamoDB 检索对话历史记录。这仅在初始连接时发生一次。

```python
def load_chat_history(userId, allowTime):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )


```

导入要放置在上下文中的历史记录并将其注册到memory_chain中。

```pytho
for item in response['Items']:
    text = item['body']['S']
    msg = item['msg']['S']
    type = item['type']['S']

    if type == 'text' and text and msg:
        memory_chain.chat_memory.add_user_message(text)
        if len(msg) > MSG_LENGTH:
            memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
        else:
            memory_chain.chat_memory.add_ai_message(msg) 
```

Serverless，比如Lambda，只有在有事件的时候才能使用，所以内存是根据事件的userId来管理的。

map_chain = dict()

```python
if userId in map_chain:  
    memory_chain = map_chain[userId]    
else: 
    memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer’,
              return_messages=True, k=10)
    map_chain[userId] = memory_chain
```

将新输入（文本）和响应（消息）保存为 user/ai 消息。

```python
memory_chain.chat_memory.add_user_message(text)
memory_chain.chat_memory.add_ai_message(msg)
```

### 使用 WebSocket 流

#### 客户端操作

要连接 WebSocket，请连接端点。您通过 onmessage() 收到一条消息。当 WebSocket 连接时，通过 onopen() 执行初始化。保活操作定期执行。如果由于网络重连等原因导致会话断开，可以通过onclose()进行检查。

```python
const ws = new WebSocket(endpoint);
ws.onmessage = function (event) {        
    response = JSON.parse(event.data)

    if(response.request_id) {
        addReceivedMessage(response.request_id, response.msg);
    }
};
ws.onopen = function () {
    isConnected = true;
    if(type == 'initial')
        setInterval(ping, 57000); 
};
ws.onclose = function () {
    isConnected = false;
    ws.close();
};
```

发送的消息为JSON格式，包括userId、请求时间、消息类型和消息内容，如下所示。发送时，使用WebSocket的send()。如果发送时会话未连接，则会显示一条通知以进行连接并重试。

```python
sendMessage({
    "user_id": userId,
    "request_id": requestId,
    "request_time": requestTime,        
    "type": "text",
    "body": message.value
})
WebSocket = connect(endpoint, 'initial');
function sendMessage(message) {
    if(!isConnected) {
        WebSocket = connect(endpoint, 'reconnect');        
        addNotifyMessage("재연결중입니다. 잠시후 다시시도하세요.");
    }
    else {
        WebSocket.send(JSON.stringify(message));     
    }     
}
```

#### 服务器操作

使用传递到 Lambda 的事件中的connectionId 和routeKey 来执行从客户端接收消息。此时会执行保活操作来维持会话。要发送消息，请在 boto3 中使用“apigatewaymanagementapi”定义客户端，然后使用 client.post_to_connection() 发送。

```python
connection_url = os.environ.get('connection_url')
client = boto3.client('apigatewaymanagementapi’,      
        endpoint_url=connection_url)

def sendMessage(id, body):
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('err_msg: ', err_msg)
        raise Exception ("Not able to send a message")

def lambda_handler(event, context):
    if event['requestContext']: 
        connectionId = event['requestContext']['connectionId']        
        routeKey = event['requestContext']['routeKey']
        
        if routeKey == '$connect':
            print('connected!')
        elif routeKey == '$disconnect':
            print('disconnected!')
        else:
            body = event.get("body", "")
            if body[0:8] == "__ping__":  # keep alive
                sendMessage(connectionId, "__pong__")
            else:
                msg, reference = getResponse(connectionId, jsonBody)
```

### 提示用法示例：翻译

使用 Prompt Engineering 轻松执行韩语/英语翻译。

```python
def translate_text(chat, text):    
    if isKorean(text)==True:        
        system = (
            "You are a helpful assistant that translates Korean to English in <article> tags. Put it in <result> tags."
        )
    else:
        system = (
            "다음의 <article> tags의 내용을 한국어로 번역하세요. 결과는 <result> tag를 붙여주세요."
        )
        
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        msg = result.content
        print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    msg = result.content
    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag
```


### 提示使用示例：修复语法错误

使用 Prompt Engineering，您可以创建一个纠正韩语/英语语法错误的 API。

```python
def check_grammer(chat, text):
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Find the error in the sentence and explain it, and add the corrected sentence at the end of your answer."
        )
        
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
```    

### 提示使用示例：总结代码

您可以使用 Prompt Engineering 创建代码摘要 API。

```python
def summary_of_code(chat, code, mode):
    if mode == 'py':
        system = (
            "다음의 <article> tag에는 python code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    elif mode == 'js':
        system = (
            "다음의 <article> tag에는 node.js code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    
    human = "<article>{code}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
```

### RAG

在 RAG 中，提示被配置为使用上下文标签包含相关文档。

```python
def query_using_RAG_context(connectionId, requestId, chat, context, revised_question):    
    system = (
            """다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.
            
            <context>
            {context}
            </context>"""
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
```

通过使用修订后的问题和使用历史记录的流来提高性能和可用性。

```python
    chain = prompt | chat
    
    stream = chain.invoke(
        {
            "context": context,
            "input": revised_question,
        }
    )
    msg = readStreamMsg(connectionId, requestId, 
            stream.content)    

    return msg
```

使用 OpenSearch 定义向量存储并注册读取的文档。

```python
def store_document_for_opensearch(bedrock_embeddings, docs, documentId):
        delete_document_if_exist(metadata_key)

        vectorstore = OpenSearchVectorSearch(
            index_name=index_name,  
            is_aoss = False,
            #engine="faiss",  # default: nmslib
            embedding_function = bedrock_embeddings,
            opensearch_url = opensearch_url,
            http_auth=(opensearch_account, opensearch_passwd),
        )
        response = vectorstore.add_documents(docs, bulk_size = 2000)
```

通过 Vectorstore 提取相关文档并将其用作上下文。

```python
# vector search (semantic) 
    relevant_documents = vectorstore_opensearch.similarity_search_with_score(
        query = query,
        k = top_k,
    )
relevant_docs = [] 
if(len(rel_docs)>=1):
        for doc in rel_docs:
            relevant_docs.append(doc)

    for document in relevant_docs:
        content = document['metadata']['excerpt']
                
        relevant_context = relevant_context + content + "\n\n"

msg = query_using_RAG_context(connectionId, requestId, chat, relevant_context, revised_question)
```

### RAG 的父/子分块

如果将文档按照大小分为父块和子块，找到子块，然后使用父块作为LLM的上下文，则可以增加搜索的准确性并使用足够的文档作为上下文。在提高RAG搜索精度的各种方法中，可以使用Parent/Child Chunking。parent-document-retrieval.md解释了父/子分块策略。

```python
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
    length_function = len,
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    # separators=["\n\n", "\n", ".", " ", ""],
    length_function = len,
)
```

执行父/子分块的过程如下。

由父/子执行分块。

当您将父文档添加到 OpenSearch 时，会创建parent_doc_id。

在子文档的元数据中注册parent_doc_id。

搜索文档时，使用过滤器搜索子文档。

如果搜索的子​​文档具有相同的父文档，则会删除重复项。

使用parent_doc_id 从 OpenSearch 检索父文档并将其用作上下文。


在父块的元数据中将“doc_level”指定为“parent”，并将其注册到 OpenSearch 中。


```python
parent_docs = parent_splitter.split_documents(docs)
    if len(parent_docs):
        for i, doc in enumerate(parent_docs):
            doc.metadata["doc_level"] = "parent"
                    
        parent_doc_ids = vectorstore.add_documents(parent_docs, bulk_size = 10000)
```

在子块的元数据中，将“doc_level”指定为“child”，将父块的文档id指定为“parent_doc_id”。

```python                
        child_docs = []
        for i, doc in enumerate(parent_docs):
            _id = parent_doc_ids[i]
            sub_docs = child_splitter.split_documents([doc])
            for _doc in sub_docs:
                _doc.metadata["parent_doc_id"] = _id
                _doc.metadata["doc_level"] = "child"
            child_docs.extend(sub_docs)
                
        child_doc_ids = vectorstore.add_documents(child_docs, bulk_size = 10000)
                    
        ids = parent_doc_ids+child_doc_ids
```

当从 OpenSearch 请求 RAG 信息时，使用 pre_filter 搜索 doc_level 为 child 的文档，如下所示。 

```python
def get_documents_from_opensearch(vectorstore_opensearch, query, top_k):
    result = vectorstore_opensearch.similarity_search_with_score(
        query = query,
        k = top_k*2,  
        pre_filter={"doc_level": {"$eq": "child"}}
    )
            
    relevant_documents = []
    docList = []
    for re in result:
        if 'parent_doc_id' in re[0].metadata:
            parent_doc_id = re[0].metadata['parent_doc_id']
            doc_level = re[0].metadata['doc_level']
```

仅当子块的parent_doc_id不重复时，才将其用作relevant_document。

```python
      
            if doc_level == 'child':
                if parent_doc_id in docList:
                    print('duplicated!')
                else:
                    relevant_documents.append(re)
                    docList.append(parent_doc_id)
                    
                    if len(relevant_documents)>=top_k:
                        break
                                
return relevant_documents
```

从 OpenSearch 导入父文档并在 RAG 中使用它。

```python
relevant_documents = get_documents_from_opensearch(vectorstore_opensearch, keyword, top_k)

for i, document in enumerate(relevant_documents):
    parent_doc_id = document[0].metadata['parent_doc_id']
    doc_level = document[0].metadata['doc_level']        
    excerpt, uri = get_parent_document(parent_doc_id) # use pareant document

def get_parent_document(parent_doc_id):
    response = os_client.get(
        index="idx-rag", 
        id = parent_doc_id
    )
    
    source = response['_source']                                
    metadata = source['metadata']    
    return source['text'], metadata['uri']
```

创建元文件在更新或删除文档时非常有用。

```python
def create_metadata(bucket, key, meta_prefix, s3_prefix, uri, category, documentId, ids):
    title = key
    timestamp = int(time.time())

    metadata = {
        "Attributes": {
            "_category": category,
            "_source_uri": uri,
            "_version": str(timestamp),
            "_language_code": "ko"
        },
        "Title": title,
        "DocumentId": documentId,      
        "ids": ids  
    }
    
    objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])

    client = boto3.client('s3')
    try: 
        client.put_object(
            Body=json.dumps(metadata), 
            Bucket=bucket, 
            Key=meta_prefix+objectName+'.metadata.json' 
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to create meta file")
```

删除或更新文档时，OpenSearch 文档将被删除。

```python
def delete_document_if_exist(metadata_key):
    try: 
        s3r = boto3.resource("s3")
        bucket = s3r.Bucket(s3_bucket)
        objs = list(bucket.objects.filter(Prefix=metadata_key))
        
        if(len(objs)>0):
            doc = s3r.Object(s3_bucket, metadata_key)
            meta = doc.get()['Body'].read().decode('utf-8')
            
            ids = json.loads(meta)['ids']

            result = vectorstore.delete(ids) 
        else:
            print('no meta file: ', metadata_key)
```

### RAG 中的文件上传

将对象上传到 S3 时发生的事件类型包括 OBJECT_CREATED_PUT（普通文件）和 CREATED_COMPLETE_MULTIPART_UPLOAD（大文件）。

```python
const s3PutEventSource = new lambdaEventSources.S3EventSource(s3Bucket, {
    events: [
      s3.EventType.OBJECT_CREATED_PUT,
      s3.EventType.OBJECT_REMOVED_DELETE,
      s3.EventType.OBJECT_CREATED_COMPLETE_MULTIPART_UPLOAD
    ],
    filters: [
      { prefix: s3_prefix+'/' },
    ]
  });
  lambdaS3eventManager.addEventSource(s3PutEventSource);
```

### 按置信度对 RAG 结果进行排序

使用 FAISS，只有超过一定可靠性水平的文档才被用作相关文档。

```python
if len(relevant_docs) >= 1:
    selected_relevant_docs = priority_search(revised_question, relevant_docs, bedrock_embeddings)

def priority_search(query, relevant_docs, bedrock_embeddings):
    excerpts = []
    for i, doc in enumerate(relevant_docs):
        excerpts.append(
            Document(
                page_content=doc['metadata']['excerpt'],
                metadata={
                    'name': doc['metadata']['title'],
                    'order':i,
                }
            )
        )  

    embeddings = bedrock_embeddings
    vectorstore_confidence = FAISS.from_documents(
        excerpts,  # documents
        embeddings  # embeddings
    )            
    rel_documents = 
        vectorstore_confidence.similarity_search_with_score(
             query=query,
             k=top_k
        )
    docs = []
    for i, document in enumerate(rel_documents):
        order = document[0].metadata['order']
        name = document[0].metadata['name']
        assessed_score = document[1]

        relevant_docs[order]['assessed_score'] = int(assessed_score)

        if assessed_score < 200:
            docs.append(relevant_docs[order])    
    return docs
```

### LangChain Agent

支持 ChatBedrock 的 Llama3，但尚不支持 Agent。相关错误如下。

```text
for chunk in self._prepare_input_and_invoke_stream(
File "/var/lang/lib/python3.11/site-packages/langchain_aws/llms/bedrock.py", line 756, in _prepare_input_and_invoke_stream
raise ValueError(
ValueError: Stop sequence key name for meta is not supported.
```

相关问题如下。

[Stop sequence key name for meta is not supported](https://github.com/langchain-ai/langchain/issues/19220)

[Error : Stop sequence key name for {meta or mistral or any other mode} is not supported](https://github.com/langchain-ai/langchain/issues/20053)

## 自己尝试一下

### 提前准备

为了使用，必须提前做好以下准备工作。

- [AWS Account](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)

### 使用CDK安装基础设施

该实验室使用俄勒冈州地区 (us-west-2)。根据基础设施安装，使用 CDK 继续进行基础设施安装。(./deployment.md). 

### 执行结果

#### 基本聊天

从菜单中选择“一般对话”，首先输入“我喜欢旅行”，然后再次输入“济州岛”，如下所示。因为我们使用了对话历史记录，所以当被问到“济州岛”时，我们会进行与济州岛旅行相关的对话。

![image](https://github.com/user-attachments/assets/8d0cd216-11e8-4d79-af62-c925808584e5)


在浏览器中选择“返回”，然后选择“4.翻译”作为对话类型，如下所示。

![image](https://github.com/kyopark2014/llama3-rag-workshop/assets/52392004/231916ba-b1e7-41ec-a8a1-dd832629b943)

庆州是韩国的一座历史名城，是新罗王朝的首都，拥有众多的文化遗产，佛国寺是被联合国教科文组织列为世界遗产的寺庙。其次，石窟庵与佛国寺一起被列为联合国教科文组织世界遗产。第三，这里是可以体验庆州历史文化遗产的地方。第四，良洞村是保留着韩国传统民居的村落。保留了原来的面貌，推荐给对历史感兴趣的人“可以参观并体验各种各样的事情”。此时的翻译结果如下。

![image](https://github.com/user-attachments/assets/dd1063a5-6d57-4754-9d99-21aee0d92254)

反过来看看英语是否可以翻译成中文, "Gyeongju is a historic city in our country. It was the capital of the Silla Kingdom and has many cultural heritages. Gyeongju has various tourist attractions. Bulguksa Temple is a UNESCO World Cultural Heritage site and has many cultural assets. This place has many Buddha statues. Second, Seokguram Grotto is a UNESCO World Cultural Heritage site along with Bulguksa Temple and has many Buddha statues. Third, it is a place where you can feel Gyeongju's historical cultural heritage. This place has Anapji Pond, Cheomseongdae Observatory, and Hwangnyongsa Temple, among others. Fourth, Yangdong Folk Village is a traditional Korean village that has preserved its old appearance. Gyeongju is recommended for those interested in history because it has many historical cultural heritages. Additionally, Gyeongju's natural scenery is also beautiful. You can have various experiences by visiting Gyeongju.". 

<img width="876" alt="image" src="https://github.com/user-attachments/assets/7a0bf9b5-0a11-41ed-ba9e-36b965bdd058">



从菜单中选择“5.语法错误纠正”。然后输入“庆州是我国的一座历史名城。它是新罗王国的首都，拥有许多文化遗产”并查看结果。不正确的语法和更正的内容如下所示。

![image](https://github.com/kyopark2014/llama3-rag-workshop/assets/52392004/9b22c400-5776-4ed5-b1cb-c551338fe053)


现在要测试 RAG，请从菜单中选择“3. RAG-opensearch（混合）”，如下所示。


![image](https://github.com/kyopark2014/llama3-rag-workshop/assets/52392004/b2daa766-a9f8-4b79-8077-a14c58e7f0f9)

[error_code.pdf](./contents/error_code.pdf)下载error_code.pdf后，选择聊天窗口中的文件图标进行上传，您可以看到文件内容摘要，如下所示。

![image](https://github.com/kyopark2014/llama3-rag-workshop/assets/52392004/5974492a-d57b-4189-bd25-7fbf7fc5b243)

现在，输入如下所示的“请详细描述锅炉错误代码”并检查结果。

![image](https://github.com/kyopark2014/llama3-rag-workshop/assets/52392004/bd740367-2d61-4d8c-9a16-6c436445a793)

如果您查看结果底部，您可以看到结果是通过 OpenSearch 的矢量/关键字搜索获得的，如下所示。

![image](https://github.com/kyopark2014/llama3-rag-workshop/assets/52392004/5ab71703-a6a8-4dfd-b406-bfa719e58259)


[ReAct-SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS](https://arxiv.org/pdf/2210.03629)选择文件图标上传，将显示汇总结果，如下所示.

![image](https://github.com/kyopark2014/llama3-rag-workshop/assets/52392004/3b1c92f7-80cd-41be-af25-c7c1a47b79f9)

现在，如果您输入 "Tell me about KNOWLEDGE-INTENSIVE REASONING TASKS"您将收到如下简短说明.

![image](https://github.com/kyopark2014/llama3-rag-workshop/assets/52392004/bbcfa84a-86ff-4cdf-a298-59adbaed0207)

要了解更多信息，您可以提出以下附加问题以获得详细信息。

![image](https://github.com/kyopark2014/llama3-rag-workshop/assets/52392004/cc7eb464-a133-41e8-9e6d-a5c11467d022)



## 整理您的资源

如果您不再使用该基础架构，您可以删除所有资源，如下所示。

1) [API Gateway Console](https://us-west-2.console.aws.amazon.com/apigateway/main/apis?region=us-west-2)访问API网关控制台， 删除“api-chatbot-for-llama3-rag-workshop”和“api-llama3-rag-workshop”。

2) [Cloud9 console](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/)访问Cloud9 控制台并使用以下命令删除所有内容。

```text
cd ~/environment/llama3.1-rag-bot/cdk-llama3-rag-workshop/ && cdk destroy --all
```




