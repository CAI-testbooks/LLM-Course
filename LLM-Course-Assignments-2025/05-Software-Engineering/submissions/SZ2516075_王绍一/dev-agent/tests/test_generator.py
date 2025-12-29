from models.code_generator import CodeGenerator


def test_code_generation():
    generator = CodeGenerator()
    
    # 测试简单的需求
    test_prompts = [
        "写一个函数，计算斐波那契数列的第n项",
        "写一个函数，检查字符串是否是回文",
        "写一个函数，对列表进行冒泡排序"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*50}")
        print(f"测试 {i+1}: {prompt}")
        print(f"{'='*50}")
        
        code = generator.generate_code(prompt)
        print(f"生成的代码：\n{code}\n")
        