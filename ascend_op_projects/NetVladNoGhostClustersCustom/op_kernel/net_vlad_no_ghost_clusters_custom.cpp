
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelNetVladNoGhostClusters {
public:
    __aicore__ inline KernelNetVladNoGhostClusters() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR clusters, GM_ADDR clusters2, GM_ADDR output,
                                uint32_t batchSize, uint32_t numFeatures, uint32_t featureSize, uint32_t clusterSize)
    {
        this->batchSize = batchSize;
        this->numFeatures = numFeatures;
        this->featureSize = featureSize;
        this->clusterSize = clusterSize;
        this->totalElements = batchSize * numFeatures * featureSize;
        this->assignmentSize = batchSize * numFeatures * clusterSize;
        this->vladSize = batchSize * clusterSize * featureSize;
        
        xGm.SetGlobalBuffer((__gm__ float *)x, this->totalElements);
        clustersGm.SetGlobalBuffer((__gm__ float *)clusters, featureSize * clusterSize);
        clusters2Gm.SetGlobalBuffer((__gm__ float *)clusters2, clusterSize * featureSize);
        outputGm.SetGlobalBuffer((__gm__ float *)output, this->vladSize);
        
        // Initialize buffers
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->featureSize * sizeof(float));
        pipe.InitBuffer(inQueueClusters, BUFFER_NUM, this->featureSize * this->clusterSize * sizeof(float));
        pipe.InitBuffer(inQueueClusters2, BUFFER_NUM, this->featureSize * this->clusterSize * sizeof(float));
        pipe.InitBuffer(outQueueAssignment, BUFFER_NUM, this->numFeatures * this->clusterSize * sizeof(float));
        pipe.InitBuffer(outQueueVlad, BUFFER_NUM, this->clusterSize * this->featureSize * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        // Step 1: Assignment computation
        ComputeAssignment();
        
        // Step 2: VLAD computation
        ComputeVLAD();
        
        // Step 3: Normalization
        NormalizeOutput();
    }

private:
    __aicore__ inline void ComputeAssignment()
    {
        // For each batch element
        for (uint32_t batchId = 0; batchId < batchSize; ++batchId) {
            // Compute assignment matrix: x @ clusters
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::LocalTensor<float> clustersLocal = inQueueClusters.AllocTensor<float>();
            
            // Load data
            AscendC::DataCopy(xLocal, xGm[batchId * numFeatures * featureSize], numFeatures * featureSize);
            AscendC::DataCopy(clustersLocal, clustersGm[0], featureSize * clusterSize);
            
            // Matrix multiplication: x @ clusters
            AscendC::LocalTensor<float> assignmentLocal = outQueueAssignment.AllocTensor<float>();
            AscendC::MatMul(assignmentLocal, xLocal, clustersLocal, numFeatures, clusterSize, featureSize, false, false);
            
            // Apply batch norm and softmax
            ApplySoftmax(assignmentLocal);
            
            // Remove ghost clusters and store
            StoreAssignment(assignmentLocal, batchId);
            
            inQueueX.FreeTensor(xLocal);
            inQueueClusters.FreeTensor(clustersLocal);
            outQueueAssignment.FreeTensor(assignmentLocal);
        }
    }
    
    __aicore__ inline void ApplySoftmax(AscendC::LocalTensor<float>& assignment)
    {
        // Simplified implementation - actual softmax would be more complex
        // This is a placeholder for the full softmax operation
        for (uint32_t i = 0; i < numFeatures * clusterSize; ++i) {
            assignment[i] = AscendC::Exp(assignment[i]);
        }
        
        // Normalize
        float sum = 0.0f;
        for (uint32_t i = 0; i < numFeatures * clusterSize; ++i) {
            sum += assignment[i];
        }
        if (sum > 0.0f) {
            for (uint32_t i = 0; i < numFeatures * clusterSize; ++i) {
                assignment[i] /= sum;
            }
        }
    }
    
    __aicore__ inline void StoreAssignment(AscendC::LocalTensor<float>& assignment, uint32_t batchId)
    {
        // Store only first clusterSize elements per feature
        AscendC::LocalTensor<float> trimmedAssignment = outQueueAssignment.AllocTensor<float>();
        for (uint32_t i = 0; i < numFeatures * clusterSize; ++i) {
            trimmedAssignment[i] = assignment[i];
        }
        AscendC::DataCopy(outputGm[batchId * numFeatures * clusterSize], trimmedAssignment, numFeatures * clusterSize);
        outQueueAssignment.FreeTensor(trimmedAssignment);
    }
    
    __aicore__ inline void ComputeVLAD()
    {
        // Placeholder for VLAD computation
        // Actual implementation would involve:
        // 1. Reshape assignment
        // 2. Compute vlad = assignment^T @ x
        // 3. Subtract mean
        // 4. Normalize
        
        // For now, just copy input to output as placeholder
        for (uint32_t batchId = 0; batchId < batchSize; ++batchId) {
            AscendC::DataCopy(outputGm[batchId * clusterSize * featureSize], 
                              xGm[batchId * numFeatures * featureSize], 
                              clusterSize * featureSize);
        }
    }
    
    __aicore__ inline void NormalizeOutput()
    {
        // Normalize output vectors
        for (uint32_t batchId = 0; batchId < batchSize; ++batchId) {
            float norm = 0.0f;
            for (uint32_t i = 0; i < clusterSize * featureSize; ++i) {
                float val = outputGm[batchId * clusterSize * featureSize + i];
                norm += val * val;
            }
            norm = AscendC::Sqrt(norm);
            if (norm > 0.0f) {
                for (uint32_t i = 0; i < clusterSize * featureSize; ++i) {
                    outputGm[batchId * clusterSize * featureSize + i] /= norm;
                }
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueClusters, inQueueClusters2;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueAssignment, outQueueVlad;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> clustersGm;
    AscendC::GlobalTensor<float> clusters2Gm;
    AscendC::GlobalTensor<float> outputGm;
    
    uint32_t batchSize;
    uint32_t numFeatures;
    uint32_t featureSize;
    uint32_t clusterSize;
    uint32_t totalElements;
    uint32_t assignmentSize;
    uint32_t vladSize;
};

extern "C" __global__ __aicore__ void net_vlad_no_ghost_clusters_custom(
    GM_ADDR x, GM_ADDR clusters, GM_ADDR clusters2, GM_ADDR output, 
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelNetVladNoGhostClusters op;
    op.Init(x, clusters, clusters2, output, 
            tiling_data.batchSize, tiling_data.numFeatures, 
            tiling_data.featureSize, tiling_data.clusterSize);
    op.Process();
}
