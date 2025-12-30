#!/usr/bin/env python3
"""
贪吃蛇游戏逻辑模块
"""

import random

class Game:
    def __init__(self, grid_width, grid_height):
        """初始化游戏状态"""
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # 初始化蛇的位置（在屏幕中央）
        start_x = grid_width // 2
        start_y = grid_height // 2
        self.snake = [(start_x, start_y), (start_x-1, start_y), (start_x-2, start_y)]
        
        # 初始方向向右
        self.direction = 'RIGHT'
        self.next_direction = 'RIGHT'
        
        # 生成第一个食物
        self.food = self.generate_food_position()
        
    def change_direction(self, new_direction):
        """改变蛇的移动方向"""
        # 防止直接反向移动（例如：向右时不能立即向左）
        opposite_directions = {
            'UP': 'DOWN',
            'DOWN': 'UP',
            'LEFT': 'RIGHT',
            'RIGHT': 'LEFT'
        }
        
        if new_direction != opposite_directions.get(self.direction):
            self.next_direction = new_direction
    
    def move(self):
        """移动蛇"""
        # 更新当前方向
        self.direction = self.next_direction
        
        # 获取蛇头位置
        head_x, head_y = self.snake[0]
        
        # 根据方向计算新的蛇头位置
        if self.direction == 'UP':
            new_head = (head_x, head_y - 1)
        elif self.direction == 'DOWN':
            new_head = (head_x, head_y + 1)
        elif self.direction == 'LEFT':
            new_head = (head_x - 1, head_y)
        elif self.direction == 'RIGHT':
            new_head = (head_x + 1, head_y)
        
        # 将新蛇头添加到蛇身前面
        self.snake.insert(0, new_head)
        
        # 移除蛇尾（除非吃到食物）
        # 注意：在main.py中，如果吃到食物会调用grow()方法，这里就不移除蛇尾
        # 所以这里总是移除蛇尾，grow()方法会额外增加长度
        self.snake.pop()
    
    def grow(self):
        """蛇增长（吃到食物时调用）"""
        # 在蛇尾添加一个新的段
        tail = self.snake[-1]
        self.snake.append(tail)
    
    def check_food_collision(self):
        """检查是否吃到食物"""
        head = self.snake[0]
        return head == self.food
    
    def check_collision(self):
        """检查碰撞（撞墙或撞到自己）"""
        head = self.snake[0]
        head_x, head_y = head
        
        # 检查是否撞墙
        if (head_x < 0 or head_x >= self.grid_width or 
            head_y < 0 or head_y >= self.grid_height):
            return True
        
        # 检查是否撞到自己（从第二个段开始检查）
        if head in self.snake[1:]:
            return True
        
        return False
    
    def generate_food(self):
        """生成新的食物位置"""
        self.food = self.generate_food_position()
    
    def generate_food_position(self):
        """生成食物位置（确保不在蛇身上）"""
        while True:
            food_x = random.randint(0, self.grid_width - 1)
            food_y = random.randint(0, self.grid_height - 1)
            food_pos = (food_x, food_y)
            
            # 确保食物不在蛇身上
            if food_pos not in self.snake:
                return food_pos