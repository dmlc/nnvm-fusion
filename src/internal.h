#ifndef NNVM_FUSION_INTERNAL_H_
#define NNVM_FUSION_INTERNAL_H_

namespace nnvm {
namespace fusion {

// TODO: will change to nnvm::subgraph
using InternalNode    = Node;
using InternalNodePtr = NodePtr;
using InternalGraph   = Graph;

} // namespace fusion
} // namespace nnvm

#endif  // NNVM_FUSION_INTERNAL_H_
