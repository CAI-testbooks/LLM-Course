#include "Simulation/FormationMission.h"

// --- 模块引用 ---
#include "Render/GeoSOTISMManager.h"
#include "Core/GeoSOTPathFinder.h"
#include "Core/GeoSOTUtils.h"
#include "CesiumGeoreference.h"

// --- JSON ---
#include "Serialization/JsonSerializer.h"
#include "JsonObjectConverter.h"

// --- 引擎工具 ---
#include "Kismet/GameplayStatics.h"
#include "DrawDebugHelpers.h"
#include "Networking.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "EngineUtils.h" 

AFormationMission::AFormationMission()
{
    PrimaryActorTick.bCanEverTick = true;

    // --- [配置] 刚性编队偏移量 ---
    float FlyH = 0.0f; // 基础高度偏移，具体高度由路径计算决定

    // 定义 2x2 编队，间距 3米 (300cm)
    // 假设中心是 (150, 150)
    DroneOffsets.Add("UAV_0", FVector(0,     0,     FlyH));
    DroneOffsets.Add("UAV_1", FVector(300.0, 0,     FlyH));
    DroneOffsets.Add("UAV_2", FVector(0,     300.0, FlyH));
    DroneOffsets.Add("UAV_3", FVector(300.0, 300.0, FlyH));
}

void AFormationMission::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // --- 实时绘制拖尾 (只画最近 5 秒的轨迹) ---
    if (!bMissionRunning) return;

    for (TActorIterator<APawn> It(GetWorld()); It; ++It)
    {
        APawn* P = *It;
        FString Name = P->GetName();
        
        // 简单匹配 AirSim 生成的 Pawn
        if (Name.Contains("FlyingPawn") || Name.Contains("UAV"))
        {
            for (auto& Elem : DroneOffsets)
            {
                FString Key = Elem.Key; 
                // 匹配 UAV_0 等后缀
                if (Name.Contains(Key.RightChop(4))) 
                {
                    FVector CurrentPos = P->GetActorLocation();
                    if (LastPositions.Contains(Key))
                    {
                        FVector LastPos = LastPositions[Key];
                        if (FVector::DistSquared(LastPos, CurrentPos) > 100.0f) 
                        {
                            // 根据 ID 分配颜色
                            int32 ID = FCString::Atoi(*Key.RightChop(4));
                            FColor Col = FLinearColor::White.ToFColor(true);
                            if (ID==0) Col=FColor::Red; 
                            else if (ID==1) Col=FColor::Green; 
                            else if (ID==2) Col=FColor::Blue; 
                            else if (ID==3) Col=FColor::Yellow;

                            DrawDebugLine(GetWorld(), LastPos, CurrentPos, Col, false, 5.0f, 0, 3.0f);
                            LastPositions[Key] = CurrentPos;
                        }
                    }
                }
            }
        }
    }
}

AActor* AFormationMission::FindActorByTag(FName Tag)
{
    TArray<AActor*> FoundActors;
    UGameplayStatics::GetAllActorsWithTag(GetWorld(), Tag, FoundActors);
    if (FoundActors.Num() > 0) return FoundActors[0];
    return nullptr;
}

// 辅助函数：解析 Tag (例如 "Urgency:High")
bool AFormationMission::ParseActorTags(AActor* Actor, FString& OutUrgency, int32& OutDeadline)
{
    if(!Actor) return false;
    
    // 默认值
    OutUrgency = "Low";
    OutDeadline = 9999;
    
    for(FName Tag : Actor->Tags)
    {
        FString S = Tag.ToString();
        
        if(S.StartsWith("Urgency:")) {
            OutUrgency = S.RightChop(8); 
        }
        else if(S.StartsWith("Deadline:")) {
            FString Val = S.RightChop(9);
            OutDeadline = FCString::Atoi(*Val);
        }
    }
    return true;
}

// ============================================================================
// AI 核心流程
// ============================================================================

void AFormationMission::RequestAIDecision()
{
    UE_LOG(LogTemp, Warning, TEXT("========== REQUESTING AI DECISION =========="));
    
    // 1. 获取引用
    if (!ISMManager) ISMManager = Cast<AGeoSOTISMManager>(UGameplayStatics::GetActorOfClass(GetWorld(), AGeoSOTISMManager::StaticClass()));
    if (!GeoRef) GeoRef = Cast<ACesiumGeoreference>(UGameplayStatics::GetActorOfClass(GetWorld(), ACesiumGeoreference::StaticClass()));

    ScanAndReportScene();
}

void AFormationMission::ScanAndReportScene()
{
    // 1. 扫描场景
    TArray<AActor*> TargetActors;
    UGameplayStatics::GetAllActorsWithTag(GetWorld(), FName("MissionTarget"), TargetActors);

    if (TargetActors.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("[UE] No actors found with tag 'MissionTarget'!"));
        return;
    }

    // 2. 构造 JSON
    FString TargetsJson = "";
    for(int i=0; i<TargetActors.Num(); ++i)
    {
        AActor* Act = TargetActors[i];
        FString Name = Act->GetName();
        FVector Loc = Act->GetActorLocation(); // cm
        
        FString Urgency;
        int32 Deadline;
        ParseActorTags(Act, Urgency, Deadline);

        TargetsJson += FString::Printf(
            TEXT("{\"id\":\"%s\", \"x\":%.1f, \"y\":%.1f, \"z\":%.1f, \"urgency\":\"%s\", \"deadline\":%d}"),
            *Name, Loc.X, Loc.Y, Loc.Z, *Urgency, Deadline
        );

        if(i < TargetActors.Num() - 1) TargetsJson += ",";
    }

    // 获取无人机当前位置 (假设以 Airport_Start 为准)
    AActor* StartNode = FindActorByTag("Airport_Start");
    FVector CurPos = StartNode ? StartNode->GetActorLocation() : FVector::ZeroVector;

    FString FinalPayload = FString::Printf(
        TEXT("{\"uav_current_pos\":{\"x\":%.1f, \"y\":%.1f, \"z\":%.1f}, \"targets\":[%s]}"),
        CurPos.X, CurPos.Y, CurPos.Z, *TargetsJson
    );

    // 3. 连接 AI Brain (Port 9999)
    ISocketSubsystem* SocketSubs = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    FSocket* Socket = SocketSubs->CreateSocket(NAME_Stream, TEXT("AIBrainLink"), false);
    TSharedRef<FInternetAddr> Addr = SocketSubs->CreateInternetAddr();
    bool bValid;
    Addr->SetIp(TEXT("127.0.0.1"), bValid);
    Addr->SetPort(9999); 

    if (Socket && Socket->Connect(*Addr))
    {
        // 发送
        TCHAR* SerializedChar = FinalPayload.GetCharArray().GetData();
        FTCHARToUTF8 Convert(SerializedChar);
        int32 Sent = 0;
        Socket->Send((uint8*)Convert.Get(), Convert.Length(), Sent);
        
        UE_LOG(LogTemp, Warning, TEXT("[UE] Scene sent to AI. Waiting for plan..."));

        // 接收 (阻塞等待)
        uint8 Buffer[20480]; 
        int32 BytesRead = 0;
        
        if (Socket->Recv(Buffer, sizeof(Buffer), BytesRead))
        {
            FString Response = FString(UTF8_TO_TCHAR((const char*)Buffer));
            Response = Response.Left(BytesRead);
            
            UE_LOG(LogTemp, Log, TEXT("[UE] AI Response Raw: %s"), *Response);

            // 解析 JSON
            TSharedPtr<FJsonObject> JsonObj;
            TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Response);

            if (FJsonSerializer::Deserialize(Reader, JsonObj))
            {
                FString Cmd = JsonObj->GetStringField("cmd");
                if (Cmd == "segmented_plan")
                {
                    FString Reason = JsonObj->GetStringField("ai_reason");
                    UE_LOG(LogTemp, Warning, TEXT("[UE] AI Reason: %s"), *Reason);

                    // 解析 segments 数组
                    TArray<TSharedPtr<FJsonValue>> SegArray = JsonObj->GetArrayField("segments");
                    TArray<FMissionSegment> MissionPlan;
                    
                    for (auto Val : SegArray)
                    {
                        TSharedPtr<FJsonObject> SegObj = Val->AsObject();
                        FMissionSegment Seg;
                        Seg.TargetID = SegObj->GetStringField("target_id");
                        Seg.Speed = SegObj->GetNumberField("speed");
                        MissionPlan.Add(Seg);
                    }

                    // 执行多段任务
                    RunMultiSegmentMission(MissionPlan);
                }
            }
        }
        Socket->Close();
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("[UE] Failed to connect to AI Brain (Port 9999)."));
    }
    
    if(Socket) SocketSubs->DestroySocket(Socket);
}

void AFormationMission::RunMultiSegmentMission(const TArray<FMissionSegment>& Segments)
{
    if (!ISMManager || !GeoRef) return;

    // 存储分段结果
    TArray<TArray<FVector>> AllSegmentsPoints;
    TArray<float> AllSpeeds;
    
    // 这里的 OccupiedSet 如果想避免“回头路”撞自己，可以一直保留
    // 这里为了简化，假设每段独立避障
    TSet<int64> GlobalOccupied; 
    
    // 渲染用的 Code
    TArray<int64> TotalVisCodes;

    // 1. 初始位置 (Airport_Start)
    AActor* StartActor = FindActorByTag("Airport_Start");
    if(!StartActor) return;
    FVector CurrentPos = StartActor->GetActorLocation();
    
    LastPositions.Empty();
    bMissionRunning = true;

    // 2. 循环每一段
    for (const FMissionSegment& Seg : Segments)
    {
        AActor* TargetActor = nullptr;
        
        // 查找 Actor
        if(Seg.TargetID == "Airport_Start") {
            TargetActor = FindActorByTag("Airport_Start");
        } else {
            TArray<AActor*> Candidates;
            UGameplayStatics::GetAllActorsWithTag(GetWorld(), FName("MissionTarget"), Candidates);
            for(AActor* A : Candidates) {
                if(A->GetName() == Seg.TargetID) { TargetActor = A; break; }
            }
        }

        if(!TargetActor) {
            UE_LOG(LogTemp, Error, TEXT("[UE] Target %s Not Found, Skipping"), *Seg.TargetID);
            continue;
        }

        FVector GoalPos = TargetActor->GetActorLocation();
        UE_LOG(LogTemp, Log, TEXT("[UE] Calculating Segment -> %s (Speed: %.1f)"), *Seg.TargetID, Seg.Speed);

        TArray<FVector> SegmentPoints;
        TArray<int64> SegmentCodes;

        // 调用寻路
        if(CalculatePathForCenter(CurrentPos, GoalPos, SegmentPoints, SegmentCodes, GlobalOccupied))
        {
            AllSegmentsPoints.Add(SegmentPoints);
            AllSpeeds.Add(Seg.Speed);
            
            TotalVisCodes.Append(SegmentCodes);
            
            // 更新下一段起点
            if(SegmentPoints.Num() > 0) {
                CurrentPos = SegmentPoints.Last();
            }
        }
    }

    // 3. 可视化
    if (ISMManager) ISMManager->DrawPath(TotalVisCodes);

    // 4. 发送给 Python AirSim (Port 8888)
    GenerateAndSendSegmentedJson(AllSegmentsPoints, AllSpeeds);
}

// ============================================================================
// 寻路与坐标核心 (已修复高度问题)
// ============================================================================

bool AFormationMission::CalculatePathForCenter(
    FVector StartPos, 
    FVector EndPos, 
    TArray<FVector>& OutPathPoints,
    TArray<int64>& OutPathCodes,
    TSet<int64>& OccupiedSet)
{
    if (!GeoRef) return false;

    int32 Level = 21; 
    double HSpan = UGeoSOTUtils::GetHeightSpan(Level);

    // 1. 获取起点的绝对海拔基准
    FVector3d StartLLH = GeoRef->TransformUnrealPositionToLongitudeLatitudeHeight(FVector3d(StartPos));
    double BaseAltitude = StartLLH.Z;

    // 计算起点网格
    int32 R1, C1;
    UGeoSOTUtils::LatLonToGeoSOT(StartLLH.Y, StartLLH.X, Level, R1, C1);
    
    // 计算起点高度层 (用于 A* 逻辑)
    int32 Alt1 = (int32)(StartLLH.Z / HSpan);
    if(Alt1 < 0) Alt1 = 0;
    int64 StartCode = UGeoSOTUtils::GenerateGeoSOTCode(R1, C1, Level, Alt1);

    // 2. 计算终点 (强制与起点同层，避免爬楼梯)
    FVector3d EndLLH = GeoRef->TransformUnrealPositionToLongitudeLatitudeHeight(FVector3d(EndPos));
    int32 R2, C2;
    UGeoSOTUtils::LatLonToGeoSOT(EndLLH.Y, EndLLH.X, Level, R2, C2);
    
    // 强制同层
    int32 Alt2 = Alt1;
    int64 EndCode = UGeoSOTUtils::GenerateGeoSOTCode(R2, C2, Level, Alt2);

    // 3. A* 寻路
    bool bFound = UGeoSOTPathFinder::FindPath(this, StartCode, EndCode, OutPathCodes, OccupiedSet);
    if (!bFound) return false;

    // 4. 还原坐标
    OutPathPoints.Empty();
    
    for (int64 Code : OutPathCodes)
    {
        int32 r, c, l, h;
        UGeoSOTUtils::DecodeGeoSOTCode(Code, r, c, l, h);
        double lat, lon;
        UGeoSOTUtils::GetGeoSOTCenter(r, c, l, lat, lon);
        
        // 核心修复：基于基准海拔计算高度
        double HeightDiff = (h - Alt1) * HSpan; 
        double FinalHeight = BaseAltitude + HeightDiff;

        FVector3d LLH(lon, lat, FinalHeight);
        FVector Pos = GeoRef->TransformLongitudeLatitudeHeightPositionToUnreal(LLH);
        OutPathPoints.Add(Pos);
    }
    return true;
}

// ============================================================================
// 通信与 JSON 生成
// ============================================================================

void AFormationMission::GenerateAndSendSegmentedJson(const TArray<TArray<FVector>>& AllSegmentsPoints, const TArray<float>& AllSpeeds)
{
    AActor* StartActor = FindActorByTag("Airport_Start");
    if(!StartActor) return;
    FVector OriginLoc = StartActor->GetActorLocation();

    FString JsonDrones = "";
    int32 SuccessCount = 0;

    // 遍历每一架无人机
    for (auto& Elem : DroneOffsets)
    {
        FString DroneName = Elem.Key;
        FVector Offset = Elem.Value;

        // 计算编队偏移 (假设中心是 150, 150)
        FVector RelativeToCenter = Offset - FVector(150, 150, 0); 

        FString SegmentsJson = "["; 

        // 遍历每一段
        for(int k=0; k < AllSegmentsPoints.Num(); ++k)
        {
            const TArray<FVector>& CurrentSegPoints = AllSegmentsPoints[k];
            float CurrentSpeed = AllSpeeds[k];

            FString PathJson = "[";
            
            // 如果是第一段，添加初始起飞点
            if(k==0) {
                float SX = Offset.X / 100.0f;
                float SY = Offset.Y / 100.0f;
                // Z = -3.0 (AirSim)
                PathJson += FString::Printf(TEXT("{\"x\":%.3f, \"y\":%.3f, \"z\":-3.0},"), SX, SY);
                
                // 记录拖尾起点
                LastPositions.Add(DroneName, OriginLoc + Offset);
            }

            for(int i=0; i<CurrentSegPoints.Num(); ++i)
            {
                // 目标 = 路径中心 + 编队偏移
                FVector TargetPos = CurrentSegPoints[i] + RelativeToCenter;
                
                // 计算相对坐标
                FVector Rel = TargetPos - OriginLoc;
                float X = Rel.X / 100.0f;
                float Y = Rel.Y / 100.0f;
                // Z轴反转 (UE Up -> AirSim Down)
                float Z = -Rel.Z / 100.0f; 
                
                PathJson += FString::Printf(TEXT("{\"x\":%.3f, \"y\":%.3f, \"z\":%.3f}"), X, Y, Z);
                if(i < CurrentSegPoints.Num()-1) PathJson += ",";
            }
            PathJson += "]";

            SegmentsJson += FString::Printf(TEXT("{\"speed\": %.1f, \"path\": %s}"), CurrentSpeed, *PathJson);
            
            if(k < AllSegmentsPoints.Num()-1) SegmentsJson += ",";
        }
        SegmentsJson += "]";

        FString DroneEntry = FString::Printf(TEXT("\"%s\": { \"segments\": %s }"), *DroneName, *SegmentsJson);
        if (SuccessCount > 0) JsonDrones += ",";
        JsonDrones += DroneEntry;
        SuccessCount++;
    }

    if (SuccessCount > 0)
    {
        // 顶层 speed 字段移除，因为 speed 已经在 segments 里了
        FString FinalJson = FString::Printf(TEXT("{ \"mission_type\": \"segmented\", \"drones\": { %s } }"), *JsonDrones);
        SendToSocket(FinalJson, 8888);
    }
}

void AFormationMission::SendToSocket(FString Payload, int32 Port)
{
    ISocketSubsystem* SocketSubs = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    FSocket* Socket = SocketSubs->CreateSocket(NAME_Stream, TEXT("SocketLink"), false);
    TSharedRef<FInternetAddr> Addr = SocketSubs->CreateInternetAddr();
    bool bValid;
    Addr->SetIp(TEXT("127.0.0.1"), bValid);
    Addr->SetPort(Port);
    
    if (Socket->Connect(*Addr))
    {
        TCHAR* SerializedChar = Payload.GetCharArray().GetData();
        FTCHARToUTF8 Convert(SerializedChar);
        int32 Sent = 0;
        Socket->Send((uint8*)Convert.Get(), Convert.Length(), Sent);
        Socket->Close();
        UE_LOG(LogTemp, Warning, TEXT("[UE] Data sent to 127.0.0.1:%d"), Port);
    }
    SocketSubs->DestroySocket(Socket);
}

// 兼容旧接口 (保留，但建议使用 RequestAIDecision)
void AFormationMission::RunFormationMission()
{
    // 如果需要手动测试单段，可以在这里写死
    // 此处留空或保留旧逻辑
    UE_LOG(LogTemp, Warning, TEXT("Please use RequestAIDecision button for AI Logic."));
}

// void AFormationMission::SendMissionToPython(FString JsonPayload)
// {
//     SendToSocket(JsonPayload, 8888);
// }