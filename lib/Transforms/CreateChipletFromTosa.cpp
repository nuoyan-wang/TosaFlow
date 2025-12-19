//===----------------------------------------------------------------------===//
// Outline TOSA ops into Chiplet chip/task.
//===----------------------------------------------------------------------===//

#include "tosa-flow/Transforms/Passes.h"
#include "tosa-flow/Dialect/Chiplet/Chiplet.h"
#include "tosa-flow/Dialect/Chiplet/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tosa_flow;
using namespace mlir::tosa_flow::chiplet;

namespace {
/// Lightweight rewriter to access the protected PatternRewriter constructor.
struct ChipletRewriter : public PatternRewriter {
  explicit ChipletRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
};
} // namespace

namespace mlir::tosa_flow {
#define GEN_PASS_DEF_CREATECHIPLETFROMTOSA
#include "tosa-flow/Transforms/Passes.h.inc"
} // namespace mlir::tosa_flow

namespace {
struct CreateChipletFromTosaPass
    : public mlir::tosa_flow::impl::CreateChipletFromTosaBase<
          CreateChipletFromTosaPass> {
  CreateChipletFromTosaPass() = default;
  explicit CreateChipletFromTosaPass(unsigned n, std::string loads = "") {
    numChiplets = n;
    chipletLoads = loads;
  }

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *ctx = func.getContext();
    unsigned chiplets = std::max<unsigned>(1, numChiplets);

    SmallVector<Operation *> targets;
    func.walk([&](Operation *op) {
      if (op->getParentOfType<TaskOp>())
        return;
      if (isa<tosa::Conv2DOp, tosa::AvgPool2dOp, tosa::MaxPool2dOp,
              tosa::MatMulOp, tosa::MulOp, tosa::SubOp, tosa::AddOp,
              tosa::DivOp, tosa::TransposeOp, tosa::ClampOp,
              tosa::ReshapeOp>(op))
        targets.push_back(op);
    });

    if (targets.empty())
      return;

    // Build per-chiplet quotas. 
    // If chiplet-loads provided, follow it
    // Otherwise, distribute evenly.
    SmallVector<unsigned> quotas;
    if (!chipletLoads.empty()) {
      llvm::StringRef spec(chipletLoads);
      SmallVector<llvm::StringRef, 8> entries;
      spec.split(entries, ',', -1, /*KeepEmpty=*/false);
      SmallVector<std::pair<unsigned, unsigned>> parsed;
      for (auto entry : entries) {
        auto trimmed = entry.trim();
        if (trimmed.empty())
          continue;
        auto pos = trimmed.find('=');
        if (pos == llvm::StringRef::npos)
          continue;
        unsigned id = 0, count = 0;
        if (!trimmed.take_front(pos).trim().getAsInteger(10, id) &&
            !trimmed.drop_front(pos + 1).trim().getAsInteger(10, count)) {
          parsed.push_back({id, count});
        }
      }
      llvm::sort(parsed, [](auto a, auto b) { return a.first < b.first; });
      for (auto [id, count] : parsed) {
        if (quotas.size() <= id)
          quotas.resize(id + 1, 0);
        quotas[id] = count;
      }
      if (!quotas.empty())
        chiplets = quotas.size();
    }

    unsigned total = targets.size();
    if (quotas.empty()) {
      unsigned base = total / chiplets;
      unsigned extra = total % chiplets;
      for (unsigned cid = 0; cid < chiplets; ++cid)
        quotas.push_back(base + (cid < extra ? 1u : 0u));
    }

    ChipletRewriter rewriter(ctx);
    unsigned idx = 0;
    while (idx < total) {
      for (unsigned cid = 0; cid < quotas.size() && idx < total; ++cid) {
        unsigned quota = quotas[cid];
        if (quota == 0)
          continue;
        unsigned chunk = std::min<unsigned>(quota, total - idx);
        ArrayRef<Operation *> slice(targets.data() + idx, chunk);
        auto task = fuseOpsIntoTask(slice, rewriter, cid);
        auto chip = task->getParentOfType<ChipOp>();
        if (chip)
          chip->setAttr("chiplet_load", rewriter.getI32IntegerAttr(quota));
        idx += chunk;
      }
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::tosa_flow::createCreateChipletFromTosaPass(
    unsigned numChiplets, std::string chipletLoads) {
  return std::make_unique<CreateChipletFromTosaPass>(numChiplets, chipletLoads);
}
