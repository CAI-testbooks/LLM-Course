/**
 * 聊天页面JavaScript
 * 支持WebSocket实时通信
 */

class WeatherChat {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.isConnected = false;
        this.autoScroll = true;
    }

    // 初始化聊天
    init(sessionId) {
        this.sessionId = sessionId;
        this.connectWebSocket();
        this.bindEvents();
    }

    // 连接WebSocket
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat/${this.sessionId}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket连接成功');
            this.isConnected = true;
            this.updateConnectionStatus('connected');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };

        this.ws.onclose = () => {
            console.log('WebSocket连接关闭');
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');

            // 尝试重新连接
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket错误:', error);
            this.updateConnectionStatus('error');
        };
    }

    // 处理消息
    handleMessage(data) {
        const messageType = data.type;

        switch (messageType) {
            case 'history':
                this.displayHistory(data.messages);
                break;

            case 'message':
                this.displayMessage(data.message);
                break;

            case 'status':
                this.updateStatus(data.content, data.progress);
                break;

            case 'system':
                this.showSystemMessage(data.content);
                break;

            case 'error':
                this.showError(data.content);
                break;
        }
    }

    // 发送消息
    sendMessage(content) {
        if (!this.isConnected || !content.trim()) return false;

        this.ws.send(JSON.stringify({
            type: 'message',
            content: content
        }));

        return true;
    }

    // 清空对话
    clearChat() {
        if (this.isConnected) {
            this.ws.send(JSON.stringify({
                type: 'clear'
            }));
        } else {
            if (confirm('确定要清空对话历史吗？')) {
                fetch(`/api/chat/${this.sessionId}/clear`, {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        this.clearChatDisplay();
                        this.showSystemMessage('对话已重置');
                    }
                });
            }
        }
    }

    // 显示历史消息
    displayHistory(messages) {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = '';

        messages.forEach(message => {
            this.displayMessage(message, false);
        });

        this.scrollToBottom();
    }

    // 显示单条消息
    displayMessage(message, animate = true) {
        const chatMessages = document.getElementById('chatMessages');
        const messageElement = this.createMessageElement(message, animate);
        chatMessages.appendChild(messageElement);

        if (this.autoScroll) {
            this.scrollToBottom();
        }
    }

    // 创建消息元素
    createMessageElement(message, animate = true) {
        const div = document.createElement('div');
        div.className = `message ${message.role}${animate ? ' animate' : ''}`;

        let avatar = '';
        let name = '';

        switch (message.role) {
            case 'user':
                avatar = '<i class="fas fa-user"></i>';
                name = '用户';
                break;
            case 'assistant':
                avatar = '<i class="fas fa-robot"></i>';
                name = '气象助手';
                break;
            case 'system':
                avatar = '<i class="fas fa-info-circle"></i>';
                name = '系统';
                break;
        }

        let metadata = '';
        if (message.metadata && message.metadata.confidence) {
            const confidence = Math.round(message.metadata.confidence * 100);
            metadata = `<div class="message-meta">
                <span class="confidence">置信度: ${confidence}%</span>
                <span class="time">${this.formatTime(message.timestamp)}</span>
            </div>`;
        } else {
            metadata = `<div class="message-meta">
                <span class="time">${this.formatTime(message.timestamp)}</span>
            </div>`;
        }

        div.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-sender">${name}</span>
                </div>
                <div class="message-text">${this.formatMessage(message.content)}</div>
                ${metadata}
            </div>
        `;

        return div;
    }

    // 格式化消息内容
    formatMessage(content) {
        // 替换换行符
        content = content.replace(/\n/g, '<br>');

        // 高亮标题
        content = content.replace(/=+\n([^\n]+)\n=+/g, (match, title) => {
            return `<div class="message-title">${title}</div>`;
        });

        // 高亮表情符号后的文本
        content = content.replace(/([🔍📊💡👥📝⚠️💡🚀🔬])\s*\*\*([^*]+)\*\*/g, '<strong>$1 $2</strong>');

        // 高亮列表项
        content = content.replace(/^\s*[\d•]\s+(.+)$/gm, '<li>$1</li>');
        content = content.replace(/(<li>.*<\/li>)/g, '<ul>$1</ul>');

        return content;
    }

    // 更新状态
    updateStatus(content, progress = 0) {
        const statusElement = document.getElementById('statusIndicator');
        const progressElement = document.getElementById('progressBar');

        if (statusElement) {
            statusElement.textContent = content;
            statusElement.style.display = 'block';

            if (progress > 0 && progressElement) {
                progressElement.style.width = `${progress}%`;
            }

            // 如果是完成状态，3秒后隐藏
            if (content === '✅ 分析完成') {
                setTimeout(() => {
                    statusElement.style.display = 'none';
                }, 3000);
            }
        }
    }

    // 显示系统消息
    showSystemMessage(content) {
        const message = {
            role: 'system',
            content: content,
            timestamp: new Date().toISOString()
        };
        this.displayMessage(message);
    }

    // 显示错误
    showError(content) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${content}`;

        const chatMessages = document.getElementById('chatMessages');
        chatMessages.appendChild(errorDiv);

        setTimeout(() => errorDiv.remove(), 5000);
    }

    // 清空聊天显示
    clearChatDisplay() {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = '';
        this.showSystemMessage('对话已重置');
    }

    // 更新连接状态
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connectionStatus');
        if (!statusElement) return;

        let icon = '';
        let text = '';
        let color = '';

        switch (status) {
            case 'connected':
                icon = 'fas fa-wifi';
                text = '已连接';
                color = '#4CAF50';
                break;
            case 'disconnected':
                icon = 'fas fa-wifi-slash';
                text = '连接断开，重连中...';
                color = '#FF9800';
                break;
            case 'error':
                icon = 'fas fa-exclamation-triangle';
                text = '连接错误';
                color = '#F44336';
                break;
        }

        statusElement.innerHTML = `<i class="${icon}"></i> ${text}`;
        statusElement.style.color = color;
    }

    // 滚动到底部
    scrollToBottom() {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // 格式化时间
    formatTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    // 绑定事件
    bindEvents() {
        // 发送消息按钮
        const sendButton = document.getElementById('sendButton');
        const messageInput = document.getElementById('messageInput');

        if (sendButton && messageInput) {
            sendButton.addEventListener('click', () => {
                const message = messageInput.value.trim();
                if (message) {
                    this.sendMessage(message);
                    messageInput.value = '';
                    messageInput.focus();
                }
            });

            // 回车发送
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendButton.click();
                }
            });
        }

        // 清空按钮
        const clearButton = document.getElementById('clearButton');
        if (clearButton) {
            clearButton.addEventListener('click', () => this.clearChat());
        }

        // 自动滚动切换
        const autoScrollToggle = document.getElementById('autoScrollToggle');
        if (autoScrollToggle) {
            autoScrollToggle.addEventListener('change', (e) => {
                this.autoScroll = e.target.checked;
            });
        }

        // 示例查询
        const exampleButtons = document.querySelectorAll('.example-button');
        exampleButtons.forEach(button => {
            button.addEventListener('click', () => {
                const example = button.dataset.example;
                if (example && messageInput) {
                    messageInput.value = example;
                    messageInput.focus();
                }
            });
        });

        // 心跳检测
        setInterval(() => {
            if (this.isConnected) {
                this.ws.send(JSON.stringify({
                    type: 'ping'
                }));
            }
        }, 30000);
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 获取会话ID
    const sessionId = document.body.dataset.sessionId;

    if (sessionId) {
        // 初始化聊天
        window.chatApp = new WeatherChat();
        window.chatApp.init(sessionId);

        // 绑定消息输入框自动调整高度
        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            });
        }

        // 示例查询点击
        const exampleQueries = document.querySelectorAll('.example-query');
        exampleQueries.forEach(item => {
            item.addEventListener('click', function() {
                const query = this.textContent;
                if (messageInput) {
                    messageInput.value = query;
                    messageInput.focus();

                    // 自动调整高度
                    messageInput.style.height = 'auto';
                    messageInput.style.height = (messageInput.scrollHeight) + 'px';
                }
            });
        });

        // 复制代码按钮
        const copyButtons = document.querySelectorAll('.copy-button');
        copyButtons.forEach(button => {
            button.addEventListener('click', function() {
                const code = this.previousElementSibling.textContent;
                navigator.clipboard.writeText(code).then(() => {
                    const originalText = this.innerHTML;
                    this.innerHTML = '<i class="fas fa-check"></i> 已复制';
                    setTimeout(() => {
                        this.innerHTML = originalText;
                    }, 2000);
                });
            });
        });
    }
});