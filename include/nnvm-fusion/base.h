/*!
 *  Copyright (c) 2016 by Contributors
 * \file base.h
 * \brief defines basic types
 */
#ifndef NNVM_FUSION_BASE_H_
#define NNVM_FUSION_BASE_H_

#include <dmlc/logging.h>
#include <nnvm/graph.h>
#include <nnvm/node.h>
#include <functional>
#include <vector>
#include <string>

#define NNVM_FUSION_DEBUG 0

namespace nnvm {
namespace fusion {

// Forward declare AST.
class AST;
// Forward declare RTC.
class RTC;

/*!
 * \brief we always use ASTPtr for a reference pointer
 * to the AST, so this alias can be changed in case.
 *
 * By default, ASTPtr is a std::shared_ptr of AST
 */
using ASTPtr        = std::shared_ptr<AST>;

/*! \brief represents generated kernel code */
using Kernel        = std::pair<std::string, std::string>;

/*! \brief The result holder of kernel of nodes in the graph */
using KernelMap     = std::unordered_map<uint32_t, Kernel>;

/*! \brief The result holder of rtc of nodes in the graph */
using RTCMap        = std::unordered_map<uint32_t, RTC>;

/*!
 * \brief Code generation function..
 *  Show how to generate ASTs for the op node with the
 *  input ASTs.
 * \param nodeptr The node to generate code.
 * \param in_asts Input ASTs of the op node.
 * \return Output ASTs of the op node.
 *
 * \note Register under "FCodeGen"
 */
using FCodeGen = std::function<std::vector<ASTPtr>(
    const NodePtr& nodeptr,
    const std::vector<ASTPtr>& in_asts)>;

} // namespace fusion
} // namespace nnvm

#endif  // NNVM_FUSION_BASE_H_
