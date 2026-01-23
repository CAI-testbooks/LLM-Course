#!/usr/bin/env python3
"""
贪吃蛇游戏主入口文件
"""

import pygame
import sys
import random
from game import Game

# 初始化pygame
pygame.init()

# 游戏常量
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE
FPS = 10

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 120, 255)
GRAY = (40, 40, 40)

class SnakeGame:
    def __init__(self):
        """初始化游戏"""
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("贪吃蛇游戏")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 24)
        
        # 游戏状态
        self.game = Game(GRID_WIDTH, GRID_HEIGHT)
        self.game_over = False
        self.paused = False
        self.score = 0
        
    def handle_events(self):
        """处理游戏事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                
                # 游戏控制
                if self.game_over:
                    if event.key == pygame.K_SPACE:
                        self.reset_game()
                else:
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_SPACE and self.paused:
                        self.paused = False
                    elif not self.paused:
                        # 方向控制
                        if event.key == pygame.K_UP:
                            self.game.change_direction('UP')
                        elif event.key == pygame.K_DOWN:
                            self.game.change_direction('DOWN')
                        elif event.key == pygame.K_LEFT:
                            self.game.change_direction('LEFT')
                        elif event.key == pygame.K_RIGHT:
                            self.game.change_direction('RIGHT')
    
    def reset_game(self):
        """重置游戏"""
        self.game = Game(GRID_WIDTH, GRID_HEIGHT)
        self.game_over = False
        self.paused = False
        self.score = 0
    
    def update(self):
        """更新游戏状态"""
        if not self.game_over and not self.paused:
            self.game.move()
            
            # 检查是否吃到食物
            if self.game.check_food_collision():
                self.game.grow()
                self.game.generate_food()
                self.score += 10
            
            # 检查游戏结束条件
            if self.game.check_collision():
                self.game_over = True
    
    def draw(self):
        """绘制游戏画面"""
        # 清屏
        self.screen.fill(BLACK)
        
        # 绘制网格背景
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (SCREEN_WIDTH, y))
        
        # 绘制蛇
        for i, segment in enumerate(self.game.snake):
            x = segment[0] * GRID_SIZE
            y = segment[1] * GRID_SIZE
            
            # 蛇头用不同颜色
            if i == 0:
                color = GREEN
            else:
                color = BLUE
                
            pygame.draw.rect(self.screen, color, (x, y, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(self.screen, BLACK, (x, y, GRID_SIZE, GRID_SIZE), 1)
        
        # 绘制食物
        food_x = self.game.food[0] * GRID_SIZE
        food_y = self.game.food[1] * GRID_SIZE
        pygame.draw.rect(self.screen, RED, (food_x, food_y, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(self.screen, WHITE, (food_x, food_y, GRID_SIZE, GRID_SIZE), 1)
        
        # 绘制分数
        score_text = self.font.render(f"分数: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # 绘制长度
        length_text = self.small_font.render(f"长度: {len(self.game.snake)}", True, WHITE)
        self.screen.blit(length_text, (10, 50))
        
        # 绘制游戏状态
        if self.game_over:
            game_over_text = self.font.render("游戏结束! 按空格键重新开始", True, RED)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            self.screen.blit(game_over_text, text_rect)
        elif self.paused:
            paused_text = self.font.render("游戏暂停 (按P继续)", True, WHITE)
            text_rect = paused_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            self.screen.blit(paused_text, text_rect)
        
        # 绘制控制说明
        controls = [
            "方向键: 控制蛇的移动",
            "P: 暂停/继续游戏",
            "空格键: 重新开始游戏",
            "ESC: 退出游戏"
        ]
        
        for i, text in enumerate(controls):
            control_text = self.small_font.render(text, True, WHITE)
            self.screen.blit(control_text, (SCREEN_WIDTH - 200, 10 + i * 25))
        
        # 更新显示
        pygame.display.flip()
    
    def run(self):
        """运行游戏主循环"""
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

if __name__ == "__main__":
    try:
        game = SnakeGame()
        game.run()
    except Exception as e:
        print(f"游戏运行出错: {e}")
        pygame.quit()
        sys.exit()