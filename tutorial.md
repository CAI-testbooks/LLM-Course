如何成为这个仓库（LLM-Course-Assignments-2025）的贡献者 

> **什么是“贡献者”？**
> 在本课程作业仓库中，“贡献者”指的是 **成功提交作业并被合并（merge）的同学**。你的作业 Pull Request 被老师或助教 review 通过并合并后，你就会正式成为这个仓库的贡献者（Contributors 列表中会出现你的 GitHub 头像和姓名）。 
> 
> 

以下是详细的完整步骤（从零开始，即使你之前没用过 GitHub 也没关系）： 

1. 注册/登录 GitHub 账号 

* 访问 [https://github.com](https://github.com) 


* 如果没有账号，点击 **Sign up** 免费注册一个（**推荐用学校邮箱**，方便以后认证学生身份享更多免费资源）。 



2. Fork 本仓库到你自己的账号下 

* 打开仓库主页（老师会提供链接，例如：`https://github.com/teacher/LLM-Course-Assignments-2025`） 


* 右上角点击 **Fork** 按钮 


* 选择 fork 到你自己的账号 


* 完成后，你会拥有一个完全属于自己的副本仓库，例如：`https://github.com/你的用户名/LLM-Course-Assignments-2025` 



3. 克隆（下载）仓库到你的电脑（推荐方式） 

* 安装 **Git**（[https://git-scm.com/downloads](https://git-scm.com/downloads)） 


* 打开终端（Windows 用 Git Bash，Mac/Linux 用 Terminal） 


* 执行以下命令： 

```bash
# 将下面的 URL 替换为你 Fork 后的仓库地址
git clone https://github.com/你的用户名/LLM-Course-Assignments-2025.git
cd LLM-Course-Assignments-2025

```



> （如果你不想本地操作，也可以用 GitHub 网页版直接编辑文件，第 5 步会介绍） 
> 
> 

4. 选择你的作业方向并准备文件 

* 阅读根目录 `README.md`，选择一个方向（例如 01-NLP）。 


* 进入对应文件夹查看其 `README.md`，了解具体题目要求。 


* 开发你的项目（代码建议放在独立仓库）。 


* **撰写最终报告**，文件名严格遵守以下格式：
`学号-姓名-方向编号-方向简称.md`
**示例：** `20231234-张三-01-NLP.md` 





5. 将报告添加到仓库 

* 方法 A：本地编辑（推荐，适合大文件或多个文件） 

  - 在对应方向的 `submissions/` 文件夹下放入你的报告文件。 
  - 在终端执行以下命令： 
```bash
git add .
git commit -m "提交最终作业报告"
git push origin main

```



* 方法 B：直接在 GitHub 网页编辑（适合只提交单个报告文件） 

  - 进入你 fork 的仓库。 


  - 导航到对应方向的 `submissions/` 文件夹。 


  - 点击 **Add file** → **Upload files**。 


  - 拖拽或选择你的报告文件上传。 


  - 下面填写 commit 信息，点击 **“Commit changes”**。 



6. 提交 Pull Request（PR）——最关键的一步 

* 回到你 fork 的仓库主页。 


* 你会看到一个黄色的提示条：“**Compare & pull request**”。 


* 点击它，或者手动点击 “Contribute” → “Open pull request”。 


* **设置标题（必须按格式）：** 示例：`[最终作业] 20231234 张三 - NLP - 医疗领域RAG问答系统` 
 
* **在描述区填写：** 


  - 你的独立代码仓库链接（**强烈推荐**） 


  - Demo 链接（如有） 


  - 主要成果和创新点 

  - 其他需要说明的内容 

* 点击 **“Create pull request”**。 



7. 等待 review 和合并 

* 老师/助教会审查你的 PR。 


* 如果需要修改，会在 PR 下留言，你可以继续推送新 commit 来更新。 


* **最终通过 review 并合并（merge）后：** 


  - 你的文件会正式出现在主仓库中。 


  - 你会自动成为本仓库的 **Contributor** （在仓库主页右侧 Contributors 列表可见）。 





---

💡 小贴士 

* 学期中可以多次提交进度版 PR（标题写 `[进度更新]`），展示你的迭代过程，**可加分**。 


* 第一次操作不熟悉很正常，多试几次就熟练了。 


* 如遇问题，可在仓库的 **Issues** 区提问，或课堂上找助教帮忙。 



**完成以上步骤，你的作业就正式提交了，同时你也成为了这个仓库的贡献者！这不仅是交作业，还是一份能放进简历的真实开源项目经验 🚀** 

加油！期待看到你的贡献～
