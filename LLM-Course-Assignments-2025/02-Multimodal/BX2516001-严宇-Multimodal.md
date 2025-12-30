我们将使用ChartQA数据集对Qwen2-VL-7B模型进行微调。该数据集包含各种图表类型的图像以及与之配对的问答对——非常适合增强模型的视觉问答能力。

📖 其他资源

如果您对更多 VLM 应用感兴趣，请查看：

多模态检索增强生成 (RAG) 方案：我将指导您使用文档检索 (ColPali) 和视觉语言模型 (VLM) 构建 RAG 系统。
Phil Schmid 的教程：深入探讨如何使用 TRL 微调多模态 LLM。
Merve Noyan 的smol-vision存储库：一系列关于前沿视觉和多模态 AI 主题的引人入胜的笔记本。
微调 VLM 图表.png

1. 安装依赖项
我们先来安装一些进行微调所必需的库！🚀

已复制已复制
 !pip install -U -q git+https://github.com/huggingface/trl.git bitsandbytes peft qwen-vl-utils trackio
 # 已使用以下版本测试：trl==0.22.0.dev0, bitsandbytes==0.47.0, peft==0.17.1, qwen-vl-utils==0.0.11, trackio==0.2.8
正在安装构建依赖项... [?25l [?25hdone
  获取构建 wheel 的要求... [?25l [?25hdone
  准备元数据 (pyproject.toml)... [?25l [?25hdone 
[2K [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ [0m [32m844.5/844.5 kB [0m [31m15.6 MB/s [0m eta [36m0:00:00 [0m 
[2K [90m–––––––––––––––––––––––––––––––––––––––[0m [32m59.6/59.6 MB [0m [31m43.7 MB/s] [0m eta] [36m0:00:00 [0m 
[2K] [90分钟━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ [0分钟 [32分钟324.6/324.6 kB [0分钟 [31分钟30.2 MB/s [0分钟预计 [36分钟0:00:00 [0分钟
[?25小时
登录 Hugging Face 上传你精心调整的模型！🗝️

您需要使用 Hugging Face 帐户进行身份验证，才能直接从此笔记本保存和分享您的模型。

已复制已复制
from huggingface_hub import notebook_login 

notebook_login()
2. 加载数据集📁
在本节中，我们将加载HuggingFaceM4/ChartQA数据集。该数据集包含图表图像以及相关的问答，非常适合用于训练视觉问答任务。

接下来，我们将为VLM生成系统消息。在本例中，我们希望创建一个系统，该系统能够像专家一样分析图表图像，并根据图像提供简洁明了的答案。

已复制已复制
system_message = """您是一个视觉语言模型，专门负责解读图表图像中的视觉数据。
您的任务是分析提供的图表图像，并用简洁的答案（通常是单个词、数字或短语）来回答查询。
图表类型多样（例如，折线图、柱状图），包含颜色、标签和文本。
请专注于根据视觉信息提供准确、简洁的答案。除非绝对必要，否则请避免额外解释。"""
我们将把数据集格式化为聊天机器人结构以进行交互。每次交互将包含系统消息、图像和用户查询，最后是查询的答案。

💡有关此型号的更多使用技巧，请查看型号卡。

已复制已复制
def  format_data ( sample ):
     return {
       "images" : [sample[ "image" ]],
       "messages" : [ 

          { "role" : "system" ,
               "content" : [ 
                  { "type" : "text" ,
                       "text" : system_message 
                  } 
              ], 
          }, 
          { "role" : "user" ,
               "content" : [ 
                  { "type" : "image" ,
                       "image" : sample[ "image" ], 
                  }, 
                  { "type" : "text" ,
                       "text" : sample[ 'query' ], 
                  } 
              ], 
          }, 
          { "role" : "assistant" ,
               "content" : [ 
                  { "type" : "text" ,
                       "text" : sample[ "label" ][ 0 ] 
                  } 
              ], 
          }, 
      ] 
      }
              
                      
              
                      
                      
              
                      
出于教学目的，我们仅加载数据集中每个分割部分的 10%。然而，在实际应用中，通常需要加载全部样本。

已复制已复制
from datasets import load_dataset 

dataset_id = "HuggingFaceM4/ChartQA" 
train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=[ 'train[:10%]' , 'val[:10%]' , 'test[:10%]' ])
我们来看一下数据集的结构。它包括一张图像、一个查询、一个标签（即答案）以及我们将要舍弃的第四个特征。

已复制已复制
训练数据集
现在，让我们使用聊天机器人结构来格式化数据。这将使我们能够为模型正确设置交互。

已复制已复制
train_dataset = [format_data(sample) for sample in train_dataset] 
eval_dataset = [format_data(sample) for sample in eval_dataset] 
test_dataset = [format_data(sample) for sample in test_dataset]
已复制已复制
train_dataset[ 200 ]
3. 加载模型并检查性能！🤔
现在我们已经加载了数据集，接下来让我们加载模型，并使用数据集中的一个样本来评估其性能。我们将使用Qwen/Qwen2-VL-7B-Instruct，这是一个能够理解视觉数据和文本的视觉语言模型 (VLM)。

如果您正在寻找替代方案，请考虑以下开源选项：

Meta AI 的Llama-3.2-11B-Vision
Mistral AI 的Pixtral-12B
Allen AI 的Molmo-7B-D-0924
此外，您还可以查看排行榜，例如WildVision Arena或OpenVLM 排行榜，以找到表现最佳的 VLM。

Qwen2_VL架构

已复制已复制
import torch
 from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor 

model_id = "Qwen/Qwen2-VL-7B-Instruct"
接下来，我们将加载模型和分词器，为推理做准备。

已复制已复制
model = Qwen2VLForConditionalGeneration.from_pretrained( 
    model_id, 
    device_map= "auto" , 
    torch_dtype=torch.bfloat16 
) 

processor = Qwen2VLProcessor.from_pretrained(model_id)
为了评估模型的性能，我们将使用数据集中的一个样本。首先，让我们看一下这个样本的内部结构。

已复制已复制
train_dataset[ 0 ]
我们将使用不包含系统消息的样本来评估 VLM 的原始理解能力。以下是我们使用的输入：

已复制已复制
train_dataset[ 0 ][ 'messages' ][ 1 : 2 ]
现在，我们来看一下与示例对应的图表。你能根据图表信息回答这个问题吗？

已复制已复制
train_dataset[ 0 ][ 'images' ][ 0 ]
我们来创建一个方法，该方法以模型、处理器和样本作为输入，生成模型的答案。这将使我们能够简化推理过程，并轻松评估虚拟逻辑模型（VLM）的性能。

已复制已复制
from qwen_vl_utils import process_vision_info def generate_text_from_sample ( model, processor, sample, max_new_tokens= 1024 , device= "cuda" ):
     # 应用聊天模板准备文本输入
    text_input = processor.apply_chat_template( 
        sample[ 'messages' ][ 1 : 2 ],   # 使用不包含系统消息的示例
        tokenize= False , 
        add_generation_prompt= True 
    ) # 处理来自示例的视觉输入
    image_inputs, _ = process_vision_info(sample[ 'messages' ]) # 为模型准备输入
    model_inputs = processor( 
        text=[text_input], 
        images=image_inputs, 
        return_tensors= "pt" , 
    ).to(device)   # 将输入移动到指定的设备# 使用模型生成文本
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens) # 修剪生成的 ID 以移除输入 ID 
    trimmed_generated_ids = [ 
        out_ids[ len (in_ids):] for in_ids, out_ids in zip (model_inputs.input_ids, generated_ids) 
    ] # 解码输出文本
    output_text = processor.batch_decode( 
        trimmed_generated_ids, 
        skip_special_tokens= True , 
        clean_up_tokenization_spaces= False 
    ) return output_text[ 0 ]   # 返回第一个解码后的输出文本

 

    

    

    

     

    

    
已复制已复制
# 如何使用示例调用该方法： 
output = generate_text_from_sample(model, processor, train_dataset[ 0 ]) 
output
虽然模型成功获取了正确的视觉信息，但它难以准确回答问题。这表明微调可能是提升其性能的关键。让我们开始微调过程吧！

移除模型并清理GPU

在下一节开始训练模型之前，让我们清除当前变量并清理 GPU 以释放资源。

已复制已复制
import gc
 import time def clear_memory ():
     # 如果变量存在于当前全局作用域中，则删除它们if 'inputs' in globals (): del globals ()[ 'inputs' ]
     if 'model' in globals (): del globals ()[ 'model' ]
     if 'processor' in globals (): del globals ()[ 'processor' ]
     if 'trainer' in globals (): del globals ()[ 'trainer' ]
     if 'bnb_config' in globals (): del globals ()[ 'bnb_config' ] 
    time.sleep( 2 ) # 垃圾回收并清除 CUDA 内存
    gc.collect() 
    time.sleep( 2 ) 
    torch.cuda.empty_cache() 
    torch.cuda.synchronize() 
    time.sleep( 2 ) 
    gc.collect() 
    time.sleep( 2 ) print ( print (f"GPU 已分配内存：{torch.cuda.memory_allocated() / 1024 ** 3 : .2 f} GB" )
     print ( f"GPU 保留内存：{torch.cuda.memory_reserved() / 1024 ** 3 : .2 f} GB" ) 
clear_memory()

 
                        

    

    
4. 使用 TRL 对模型进行微调
4.1 加载量化模型进行训练 ⚙️
接下来，我们将使用bitsandbytes加载量化模型。如果您想了解更多关于量化的信息，请查看这篇博文或这篇博文。

已复制已复制
from transformers import BitsAndBytesConfig # BitsAndBytesConfig int-4 配置
bnb_config = BitsAndBytesConfig( 
    load_in_4bit= True , 
    bnb_4bit_use_double_quant= True , 
    bnb_4bit_quant_type= "nf4" , 
    bnb_4bit_compute_dtype=torch.bfloat16 
) # 加载模型和分词器
model = Qwen2VLForConditionalGeneration.from_pretrained( 
    model_id, 
    device_map= "auto" , 
    torch_dtype=torch.bfloat16, 
    quantization_config=bnb_config 
) 
processor = Qwen2VLProcessor.from_pretrained(model_id)



4.2 设置 QLoRA 和 SFTConfig 🚀
接下来，我们将配置QLoRA以用于训练设置。与传统方法相比，QLoRA 能够高效地微调大型语言模型，同时显著降低内存占用。与通过应用低秩近似来降低内存使用的标准 LoRA 不同，QLoRA 更进一步，通过量化 LoRA 适配器的权重来实现这一点。这进一步降低了内存需求并提高了训练效率，使其成为在不牺牲模型质量的前提下优化模型性能的绝佳选择。

已复制已复制
from peft import LoraConfig # 配置 LoRa 
peft_config = LoraConfig( 
    lora_alpha= 16 , 
    lora_dropout= 0.05 , 
    r= 8 , 
    bias= "none" , 
    target_modules=[ "q_proj" , "v_proj" ], 
    task_type= "CAUSAL_LM" , 
)

我们将使用监督式微调 (SFT) 来提升模型在当前任务上的性能。为此，我们将使用TRL 库中的SFTConfig类来定义训练参数。SFT 允许我们提供带标签的数据，帮助模型学习根据接收到的输入生成更准确的响应。这种方法确保模型能够适应我们的特定用例，从而在理解和响应视觉查询方面获得更好的性能。

已复制已复制
from trl import SFTConfig # 配置训练参数
training_args = SFTConfig( 
    output_dir= "qwen2-7b-instruct-trl-sft-ChartQA" ,   # 模型保存目录
    num_train_epochs= 3 ,   # 训练轮数
    per_device_train_batch_size= 4 ,   # 训练批次大小
    per_device_eval_batch_size= 4 ,   # 评估批次大小
    gradient_accumulation_steps= 8 ,   # 梯度累积步数
    gradient_checkpointing_kwargs={ "use_reentrant" : False },   # 梯度检查点选项
    max_length= None ,
     # 优化器和调度器设置
    optim= "adamw_torch_fused" ,   # 优化器类型
    learning_rate= 2e-4 ,   # 训练学习率# 日志记录和评估
    logging_steps= 10 ,   # 日志记录间隔
    eval_steps= 10 ,   # 评估步数间隔
    eval_strategy= "steps" ,   # 评估策略
    save_strategy= "steps" ,   # 模型保存策略
    save_steps= 20 ,   # 保存步数间隔# 混合精度和梯度设置
    bf16= True ,   # 使用 bfloat16 精度
    max_grad_norm= 0.3 ,   # 梯度裁剪的最大范数
    warmup_ratio= 0.03 ,   # 预热步数占总步数的比例# Hub 和报告
    push_to_hub= True ,   # 是否将模型推送到 Hugging Face Hub 
    report_to= "trackio" 
, #  用于跟踪指标的报告工具


    
    
    
4.3 训练模型🏃
我们将使用Trackio 记录训练进度。让我们将笔记本电脑连接到 W&B，以便在训练期间捕获关键信息。

已复制已复制
import trackio  trackio.init(
      project= "qwen2-7b-instruct-trl-sft-ChartQA" ,
      name= "qwen2-7b-instruct-trl-sft-ChartQA" ,
      config=training_args,
      space_id=training_args.output_dir + "-trackio"  )


* Trackio 项目已初始化：qwen2-7b-instruct-trl-sft-ChartQA 
* Trackio 指标将同步到 Hugging Face 数据集：sergiopaniego/qwen2-7b-instruct-trl-sft-ChartQA-trackio-dataset 
* 创建新空间：https://huggingface.co/spaces/sergiopaniego/qwen2-7b-instruct-trl-sft-ChartQA-trackio 
* 访问以下链接查看仪表盘：https://huggingface.co/spaces/sergiopaniego/qwen2-7b-instruct-trl-sft-ChartQA-trackio
现在，我们将定义SFTTrainer ，它是transformers.Trainer类的封装，并继承了其属性和方法。当提供PeftConfig对象时，该类会正确初始化PeftModel，从而简化微调过程。通过使用 SFTTrainer ，我们可以高效地管理训练工作流程，并确保视觉语言模型获得流畅的微调体验。在进行推理时，我们定义了自己的函数，该函数在将输入传递给模型之前应用必要的预处理。在这里，SFTTrainer 会自动推断该模型是一个视觉语言模型，并应用一个将输入转换为适当格式的转换函数。SFTTrainergenerate_text_from_sampleDataCollatorForVisionLanguageModeling

已复制已复制
from trl import SFTTrainer 

trainer = SFTTrainer( 
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset, 
    peft_config=peft_config, 
    processing_class=processor, 
)
是时候训练模型了！🎉

已复制已复制
训练器.训练()
让我们保存结果吧💾

已复制已复制
trainer.save_model(training_args.output_dir)
5. 测试微调后的模型🔍
现在我们已经成功微调了视觉语言模型 (VLM)，是时候评估它的性能了！在本节中，我们将使用 ChartQA 数据集中的示例来测试模型，看看它如何回答基于图表图像的问题。让我们深入了解一下结果吧！🚀

让我们清理一下GPU内存，以确保最佳性能🧹

已复制已复制
清除内存()
我们将使用与之前相同的流程重新加载基础模型。

已复制已复制
model = Qwen2VLForConditionalGeneration.from_pretrained( 
    model_id, 
    device_map= "auto" , 
    torch_dtype=torch.bfloat16 
) 

processor = Qwen2VLProcessor.from_pretrained(model_id)
我们将把训练好的适配器附加到预训练模型上。该适配器包含了我们在训练过程中所做的微调，使基础模型能够在不改变其核心参数的情况下利用这些新知识。通过集成该适配器，我们可以在保持模型原有结构的同时增强其性能。

已复制已复制
adapter_path = "sergiopaniego/qwen2-7b-instruct-trl-sft-ChartQA" 
model.load_adapter(adapter_path)
我们将利用模型最初难以正确回答的数据集中的先前样本。

已复制已复制
train_dataset[ 0 ][ 'messages' ][: 2 ]
已复制已复制
 train_dataset[ 0 ][ 'images' ][ 0 ]

已复制已复制
output = generate_text_from_sample(model, processor, train_dataset[ 0 ]) 
output
由于该样本取自训练集，模型在训练过程中已经遇到过它，这可能被视为一种作弊行为。为了更全面地了解模型的性能，我们还将使用一个未见过的样本对其进行评估。

已复制已复制
test_dataset[ 10 ][ 'messages' ][: 2 ]
已复制已复制
 test_dataset[ 10 ][ 'images' ][ 0 ]

已复制已复制
output = generate_text_from_sample(model, processor, test_dataset[ 10 ]) 
output
模型已成功学习并能响应数据集中指定的查询。我们达成目标啦！🎉✨

💻 我开发了一个示例应用程序来测试该模型，您可以在这里找到它。您可以轻松地将其与另一个包含预训练模型的空间进行比较，该空间可在此处获取。

已复制已复制
from IPython.display import IFrame 

IFrame(src= "https://sergiopaniego-qwen2-vl-7b-trl-sft-chartqa.hf.space" , width= 1000 , height= 800 )
6. 对比微调模型与基础模型+提示 📊
我们已经探讨了如何通过微调 VLM 来使其适应我们的特定需求。另一种值得考虑的方法是直接使用提示或实施 RAG 系统，这将在另一篇文章中介绍。

对虚拟逻辑模型 (VLM) 进行微调需要大量数据和计算资源，这会产生费用。相比之下，我们可以尝试使用提示功能，看看能否在无需微调的情况下获得类似的结果。

让我们再次清理GPU内存，以确保最佳性能🧹

已复制已复制
 clear_memory()
GPU分配显存：0.02 GB；
GPU保留显存：0.27 GB
🏗️ 首先，我们将按照与之前相同的流程加载基线模型。

已复制已复制
model = Qwen2VLForConditionalGeneration.from_pretrained( 
    model_id, 
    device_map= "auto" , 
    torch_dtype=torch.bfloat16 
) 

processor = Qwen2VLProcessor.from_pretrained(model_id)
📜 在这种情况下，我们将再次使用之前的示例，但这次我们将包含如下系统消息。添加此消息有助于为模型提供上下文信息，从而可能提高其响应准确率。

已复制已复制
train_dataset[ 0 ][: 2 ]
让我们看看它的表现如何！

已复制已复制
text = processor.apply_chat_template( 
    train_dataset[ 0 ][: 2 ], tokenize= False , add_generation_prompt= True 
) 

image_inputs, _ = process_vision_info(train_dataset[ 0 ]) 

inputs = processor( 
    text=[text], 
    images=image_inputs, 
    return_tensors= "pt" , 
) 

inputs = inputs.to( "cuda" ) 

generated_ids = model.generate(**inputs, max_new_tokens= 1024 ) 
generated_ids_trimmed = [out_ids[ len (in_ids):] for in_ids, out_ids in  zip (inputs.input_ids, generated_ids)] 

output_text = processor.batch_decode( 
    generated_ids_trimmed, 
    skip_special_tokens= True , 
    clean_up_tokenization_spaces= False 
) 

output_text[ 0 ]
💡 正如我们所见，该模型无需任何训练，即可利用预训练模型和额外的系统消息生成正确答案。根据具体应用场景，这种方法或许可以作为微调的一种可行替代方案。
