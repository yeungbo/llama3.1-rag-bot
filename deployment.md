# 安装基础设施

## 设置模型权限

在这里，要使用多地区LLM，请访问下面的链接，选择[编辑]，并将所有模型设置为可用。特别是，“Llama 3.1 70B Instruct”、“Titan Embeddings G1 - Text”和“Titan Text Embedding V2”必须可用于 LLM 和矢量嵌入。

- [Model access - Oregon](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/modelaccess)

   

## 使用 CDK 安装基础设施


在这里，我们使用Cloud9中的AWS CDK安装基础设施。

1) [Cloud9 Console](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/create)在【创建环境】-【名称】中输入名称“chatbot”，EC2实例选择“m5.large”。将其余部分保留为默认值，滚动到底部并选择“创建”。

![noname](https://github.com/kyopark2014/chatbot-based-on-Falcon-FM/assets/52392004/7c20d80c-52fc-4d18-b673-bd85e2660850)

2) [Environment](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/)在环境中【打开】“chatbot”后，运行终端，如下图所示。

![noname](https://github.com/kyopark2014/chatbot-based-on-Falcon-FM/assets/52392004/b7d0c3c0-3e94-4126-b28d-d269d2635239)

3) 更改EBS大小

下载脚本，如下所示。

```text
curl https://raw.githubusercontent.com/kyopark2014/technical-summary/main/resize.sh -o resize.sh
```

然后，使用以下命令将容量更改为 80G。
```text
chmod a+rx resize.sh && ./resize.sh 80
```


4) 下载源代码。

```java
git clone https://github.com/kyopark2014/llama3-rag-workshop
```

5) 转到 cdk 文件夹并安装所需的库。

```java
cd llama3.1-rag-bot/cdk-llama3-rag-workshop/ && npm install
```

6) 执行引导以使用 CDK。

使用以下命令检查您的帐户 ID。

```java
aws sts get-caller-identity --query Account --output text
```

执行引导程序，如下所示。这里，“account-id”是通过上述命令确认的 12 位帐户 ID。您只需要运行 bootstrap 一次，因此如果您已经在使用 cdk，则可以跳过 bootstrap。

```java
cdk bootstrap aws://[account-id]/us-west-2
```

7)安装基础设施。

```java
cdk deploy --all
```

安装完成后，将出现以下输出。

![noname](https://github.com/kyopark2014/llama3-rag-workshop/assets/52392004/33c22399-ea95-4602-a018-fb6b280cec7b)


8) 将 HTML 文件复制到 S3。

粘贴 Output 的 HtmlUpdateCommend，如下所示。

![noname](https://github.com/kyopark2014/llama3-rag-workshop/assets/52392004/103ad54b-6628-4135-80d9-15a33ac2d384)




9) 安装 Nori 插件以进行混合搜索

[OpenSearch Console](https://us-west-2.console.aws.amazon.com/aos/home?region=us-west-2#opensearch/domains)에서 "llama3.1-rag-bot"로 들어가서 [Packages] - [Associate package]在OpenSearch Console, 中进入“llama3.1-rag-bot” ，选择[Packages] - [Associate package]，然后安装“analysis-nori”，如下所示。

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/b91c91a1-b13c-4f5d-bd58-1c8298b2f128)

11) 从输出中复制 WebUrlforllama3ragworkshop 并连接到浏览器。
