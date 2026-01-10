#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "FormationMission.generated.h"

// 前向声明
class AGeoSOTISMManager;
class ACesiumGeoreference;

// 定义任务片段结构体
struct FMissionSegment {
	FString TargetID;
	float Speed;
};

UCLASS()
class GEOSOTSYSTEM_API AFormationMission : public AActor
{
	GENERATED_BODY()
    
public:    
	AFormationMission();
	virtual void Tick(float DeltaTime) override;

	// --- [旧入口] 单次任务 (保留用于调试) ---
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "GeoSOT|Mission")
	void RunFormationMission();
    
	// --- [新入口] 点击此按钮开始 AI 智能任务 ---
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "GeoSOT|AI")
	void RequestAIDecision();

private:
	// 1. 扫描场景并发送给 AI Brain (Port 9999)
	void ScanAndReportScene();
    
	// 2. 执行多段任务链 (AI 返回结果后调用)
	void RunMultiSegmentMission(const TArray<FMissionSegment>& Segments);

	// 3. 生成最终给 AirSim 的分段 JSON 并发送 (Port 8888)
	void GenerateAndSendSegmentedJson(const TArray<TArray<FVector>>& AllSegmentsPoints, const TArray<float>& AllSpeeds);

	// 辅助：解析 Tags 获取元数据 (Urgency, Deadline)
	bool ParseActorTags(AActor* Actor, FString& OutUrgency, int32& OutDeadline);
    
	// 辅助：基础寻路 (包含高度修复逻辑)
	bool CalculatePathForCenter(FVector StartPos, FVector EndPos, TArray<FVector>& OutPathPoints, TArray<int64>& OutPathCodes, TSet<int64>& OccupiedSet);
    
	// 辅助：通用 Socket 发送
	void SendToSocket(FString Payload, int32 Port);
    
	// 辅助：查找 Actor
	AActor* FindActorByTag(FName Tag);

private:
	// 偏移量配置 (UAV_0, UAV_1...)
	TMap<FString, FVector> DroneOffsets;

	// 引用
	UPROPERTY()
	AGeoSOTISMManager* ISMManager;
    
	UPROPERTY()
	ACesiumGeoreference* GeoRef;

	// 轨迹绘制相关
	TMap<FString, FVector> LastPositions;
	bool bMissionRunning = false;
};