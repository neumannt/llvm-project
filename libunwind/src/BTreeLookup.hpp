//===-BtreeLookup.hpp ----------------.....--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Organize unwinding information in a btree for contention free lookup.
// Has to be explicitly enabled by the application, and shared library must
// be registered after loading and de-registered before unloading.
//
//===----------------------------------------------------------------------===//

#ifndef __BTREE_LOOKUP_HPP__
#define __BTREE_LOOKUP_HPP__

#include "config.h"
#include "RWMutex.hpp"
#include <limits.h>

namespace libunwind {

// A b+-tree structure that uses optimistic lock coupling for contention free
// unwinding. This is explicitly opt-in, as it has to be informed about shared
// library that have been loaded or that are about to be removed.
// All methods are thread safe
//
// Internally, this data structure uses two different btrees, as we have to
// treat unwind tables from shared libraries different than explicitly registered
// unwind frames. We use Node<>/Tree<> templates to avoid code duplication.
class _LIBUNWIND_HIDDEN BTreeLookup {
private:
  // The largest possible separator
  static constexpr uintptr_t maxSeparator = ~static_cast<uintptr_t>(0);

  // Common logic for version locks
  struct VersionLock {
    // The version lock itself
    uintptr_t versionLock;

    // Initialize in locked state
    void initializeLockedExclusive() { versionLock = 1; }

    // Try to lock the node exclusive
    bool tryLockExclusive();
    // Lock the node exclusive, blocking as needed
    void lockExclusive();
    // Release a locked node and increase the version lock
    void unlockExclusive();
    // Acquire an optimistic "lock". Note that this does not lock at all, it only allows for validation later
    bool lockOptimistic(uintptr_t &lock);
    // Validate a previously acquire lock
    bool validate(uintptr_t lock);
  };
  // A btree node
  template <class Payload>
  struct Node {
    // Inner entry. The child tree contains all entries < separator
    struct InnerEntry {
      uintptr_t separator;
      Node *child;
    };
    // Leaf entry. Depending on the use case we store a different payload here
    struct LeafEntry {
      uintptr_t base, size;
      Payload payload;
    };
    static constexpr unsigned desiredNodeSize = 256 - 16;
    static constexpr unsigned maxFanoutInner = (desiredNodeSize / sizeof(InnerEntry));
    static constexpr unsigned maxFanoutLeaf = (desiredNodeSize / sizeof(LeafEntry));
    static_assert((maxFanoutInner > 4) && (maxFanoutLeaf > 4), "node size too small");

    // The version lock used for optimistic lock coupling
    VersionLock versionLock;
    // The number of entries
    unsigned entryCount;
    // The type
    enum { Inner,
           Leaf,
           Free } type;
    // The payload
    union {
      // The inner nodes have fence keys, i.e., the right-most entry includes a separator
      InnerEntry children[maxFanoutInner + 1];
      LeafEntry entries[maxFanoutLeaf];
    } content;

    // Is an inner node?
    bool isInner() const { return type == Inner; }
    // Is a leaf node?
    bool isLeaf() const { return type == Leaf; }
    // Should the node be merged?
    bool needsMerge() const { return entryCount < (isInner() ? (maxFanoutInner / 2) : (maxFanoutLeaf / 2)); }
    // Get the fence key for inner nodes
    uintptr_t getFenceKey() const;

    // Find the position for a slot in an inner node
    unsigned findInnerSlot(uintptr_t value) const;
    // Find the position for a slot in a leaf node
    unsigned findLeafSlot(uintptr_t value) const;

    // Try to lock the node exclusive
    bool tryLockExclusive() { return versionLock.tryLockExclusive(); }
    // Lock the node exclusive, blocking as needed
    void lockExclusive() { versionLock.lockExclusive(); }
    // Release a locked node and increase the version lock
    void unlockExclusive() { versionLock.unlockExclusive(); }
    // Acquire an optimistic "lock". Note that this does not lock at all, it only allows for validation later
    bool lockOptimistic(uintptr_t &lock) { return versionLock.lockOptimistic(lock); }
    // Validate a previously acquire lock
    bool validate(uintptr_t lock) { return versionLock.validate(lock); }

    // Insert a new separator after splitting
    void updateSeparatorAfterSplit(uintptr_t oldSeparator, uintptr_t newSeparator, Node *newRight);
  };
  // A btree
  template <class Payload>
  struct Tree {
    using Node = BTreeLookup::Node<Payload>;

    // The root of the btree
    Node *root = nullptr;
    // The free list of released node
    Node *freeList = nullptr;
    // The version lock used to protect the root
    VersionLock rootLock = {0};

    ~Tree();

    // Allocate a node. This node will be returned in locked exclusive state
    Node *allocateNode(bool inner);
    // Release a node. This node must be currently locked exclusively and will be placed in the free list
    void releaseNode(Node *node);
    // Recursive release a tree
    void releaseTreeRecursively(Node *node);

    // Check if we are splitting the root
    void handleRootSplit(Node *&node, Node *&parent);
    // Split an inner node
    void splitInner(Node *&inner, Node *&parent, uintptr_t target);
    // Split a leaf node
    void splitLeaf(Node *&leaf, Node *&parent, uintptr_t fence, uintptr_t target);
    // Merge (or balance) child nodes
    Node *mergeNode(unsigned childSlot, Node *parent, uintptr_t target);

    // Does the tree have a root
    bool hasRoot() const { return __atomic_load_n(&root, __ATOMIC_SEQ_CST); }

    // Insert an entry
    bool insert(uintptr_t base, uintptr_t size, Payload payload, Node **secondaryRoot = nullptr);
    // Remove an entry
    bool remove(uintptr_t base);

    // Lookup result
    struct LookupResult {
      uintptr_t base, size;
      Payload payload;
    };
    // Find the corresponding entry the given address
    bool lookup(uintptr_t targetAddr, LookupResult &result);
  };
  // Lookup data for EH tables
  struct EHTableInfo {
    uintptr_t dwarf_section;
    size_t dwarf_section_length;
    uintptr_t dwarf_index_section;
    size_t dwarf_index_section_length;
  };
  // Lookup data for explicitly registered frames
  struct ExplicitFrameInfo {
    uintptr_t fde;
  };
  // The tree for table lookup
  Tree<EHTableInfo> tableTree;
  // The tree for explicit frames
  Tree<ExplicitFrameInfo> frameTree;
  // Protection against concurrent sync calls
  RWMutex mutex;
  // A spinlock for initialization. This will be used only once
  uint8_t initialization = 0;

  // Info for the callback
  struct CallbackInfo {
    BTreeLookup *lookup;
    Node<EHTableInfo> **newRoot;
  };
  // Callback for dl_iterate_phdr
  static int iterateCallback(struct dl_phdr_info *info, size_t size, void *data);

public:
  BTreeLookup();
  ~BTreeLookup();

  // Is the object properly constructed? Used to handle early calls at startup
  bool isConstructed() const { return __atomic_load_n(&initialization, __ATOMIC_SEQ_CST) > 0; }
  // Is the cache enabled?
  bool isEnabled() const { return tableTree.hasRoot() || frameTree.hasRoot(); }

  // Synchronize the btree. Has to be called once at startup and once after ever dlopen/dlclose
  void sync();

  // Explicitly register a frame
  bool insertFrame(uintptr_t base, uintptr_t size, uintptr_t fde);
  // Remove a previously registered frame
  bool removeFrame(uintptr_t base);

  // Find the corresponding unwinding info for the given address
  uintptr_t findFDE(uintptr_t targetAddr);
};

bool BTreeLookup::VersionLock::tryLockExclusive()
// Try to lock the lock exclusive
{
  uintptr_t state = __atomic_load_n(&versionLock, __ATOMIC_SEQ_CST);
  if (state & 1)
    return false;
  return __atomic_compare_exchange_n(&versionLock, &state, state | 1, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

void BTreeLookup::VersionLock::lockExclusive()
// Lock the lock exclusive, blocking as needed
{
  // We should virtually never get contention here, as nodes are only
  // modified after dlopen/dlclose calls. Thus we use a simple spinlock
  while (true) {
    uintptr_t state = __atomic_load_n(&versionLock, __ATOMIC_SEQ_CST);
    if (state & 1)
      continue;
    if (__atomic_compare_exchange_n(&versionLock, &state, state | 1, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
      return;
  }
}

void BTreeLookup::VersionLock::unlockExclusive()
// Release a locked node and increase the version lock
{
  uintptr_t state = __atomic_load_n(&versionLock, __ATOMIC_SEQ_CST);
  assert(state & 1);
  __atomic_store_n(&versionLock, (state + 2) & (~static_cast<uintptr_t>(1)), __ATOMIC_SEQ_CST);
}

bool BTreeLookup::VersionLock::lockOptimistic(uintptr_t &lock)
// Acquire an optimistic "lock". Note that this does not lock at all, it only allows for validation later
{
  uintptr_t state = __atomic_load_n(&versionLock, __ATOMIC_SEQ_CST);
  lock = state;

  // Acquire the lock fails when there is currently an exclusive lock
  return !(state & 1);
}

bool BTreeLookup::VersionLock::validate(uintptr_t lock)
// Validate a previously acquire lock
{
  // Check that the node is still in the same state
  uintptr_t state = __atomic_load_n(&versionLock, __ATOMIC_SEQ_CST);
  return (state == lock);
}

template <class Payload>
uintptr_t BTreeLookup::Node<Payload>::getFenceKey() const
// Get the fence key for inner nodes
{
  // We only ask this for non-empty nodes
  assert((entryCount > 0) && isInner());

  // For inner nodes we just return our right-most entry
  return content.children[entryCount - 1].separator;
}

template <class Payload>
unsigned BTreeLookup::Node<Payload>::findInnerSlot(uintptr_t value) const
// The the position for a slot in an inner node
{
  for (unsigned index = 0; index != entryCount; ++index)
    if (content.children[index].separator >= value)
      return index;
  return entryCount;
}

template <class Payload>
unsigned BTreeLookup::Node<Payload>::findLeafSlot(uintptr_t value) const
// The the position for a slot in an inner node
{
  for (unsigned index = 0; index != entryCount; ++index)
    if (content.entries[index].base + content.entries[index].size > value)
      return index;
  return entryCount;
}

template <class Payload>
void BTreeLookup::Node<Payload>::updateSeparatorAfterSplit(uintptr_t oldSeparator, uintptr_t newSeparator, Node *newRight)
// Insert a new separator after splitting
{
  assert(entryCount < maxFanoutInner);

  unsigned slot = findInnerSlot(oldSeparator);
  assert((slot < entryCount) && ((content.children[slot].separator == oldSeparator) || (((slot + 1) == entryCount) && (content.children[slot].separator == maxSeparator))));
  for (unsigned index = entryCount; index > slot; --index)
    content.children[index] = content.children[index - 1];
  content.children[slot].separator = newSeparator;
  content.children[slot + 1].child = newRight;
  ++entryCount;
}

template <class Payload>
BTreeLookup::Tree<Payload>::~Tree() {
  // Disable the mechanism before cleaning up
  Node *oldRoot = __atomic_exchange_n(&root, nullptr, __ATOMIC_SEQ_CST);
  if (oldRoot)
    releaseTreeRecursively(oldRoot);

  // Release all free pages
  while (freeList) {
    Node *next = freeList->content.children[0].child;
    free(freeList);
    freeList = next;
  }
}

// Allocate a node. This node will be returned in locked exclusive state
template <class Payload>
BTreeLookup::Node<Payload> *BTreeLookup::Tree<Payload>::allocateNode(bool inner) {
  while (true) {
    // Try the free list first
    Node *nextFree = __atomic_load_n(&freeList, __ATOMIC_SEQ_CST);
    if (nextFree) {
      if (!nextFree->tryLockExclusive())
        continue;
      // The node might no longer be free, check that again after acquiring the exclusive lock
      if (nextFree->type == Node::Free) {
        Node *ex = nextFree;
        if (__atomic_compare_exchange_n(&freeList, &ex, nextFree->content.children[0].child, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
          nextFree->entryCount = 0;
          nextFree->type = inner ? Node::Inner : Node::Leaf;
          return nextFree;
        }
      }
      nextFree->unlockExclusive();
      continue;
    }

    // No free page available, allocate a new one
    Node *newPage = static_cast<Node *>(malloc(sizeof(Node)));
    newPage->versionLock.initializeLockedExclusive(); // initialize the node in locked state
    newPage->entryCount = 0;
    newPage->type = inner ? Node::Inner : Node::Leaf;
    return newPage;
  }
}

template <class Payload>
void BTreeLookup::Tree<Payload>::releaseNode(Node *node) {
  // We cannot release the memory immediately because there might still be
  // concurrent readers on that node. Put it in the free list instead
  node->type = Node::Free;
  Node *nextFree = __atomic_load_n(&freeList, __ATOMIC_SEQ_CST);
  do {
    node->content.children[0].child = nextFree;
  } while (!__atomic_compare_exchange_n(&freeList, &nextFree, node, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
  node->unlockExclusive();
}

// Recursive release a tree. The btree is by design very shallow, thus
// we can risk recursion here
template <class Payload>
void BTreeLookup::Tree<Payload>::releaseTreeRecursively(Node *node) {
  node->lockExclusive();
  if (node->isInner()) {
    for (unsigned index = 0; index < node->entryCount; ++index)
      releaseTreeRecursively(node->content.children[index].child);
  }
  releaseNode(node);
}

template <class Payload>
void BTreeLookup::Tree<Payload>::handleRootSplit(Node *&node, Node *&parent) {
  // We want to keep the root pointer stable to allow for contention
  // free reads. Thus, we split the root by first moving the content
  // of the root node to a new node, and then split that new node
  if (!parent) {
    // Allocate a new node, we guarantees us that we will have a parent afterwards
    Node *newNode = allocateNode(node->isInner());
    newNode->entryCount = node->entryCount;
    newNode->content = node->content;
    node->content.children[0].separator = maxSeparator;
    node->content.children[0].child = newNode;
    node->entryCount = 1;
    node->type = Node::Inner;

    parent = node;
    node = newNode;
  }
}

template <class Payload>
void BTreeLookup::Tree<Payload>::splitInner(Node *&inner, Node *&parent, uintptr_t target)
// Split an inner node
{
  // Check for the root
  handleRootSplit(inner, parent);

  // Create two inner node
  uintptr_t rightFence = inner->getFenceKey();
  Node *leftInner = inner;
  Node *rightInner = allocateNode(true);
  unsigned split = leftInner->entryCount / 2;
  rightInner->entryCount = leftInner->entryCount - split;
  for (unsigned index = 0; index < rightInner->entryCount; ++index)
    rightInner->content.children[index] = leftInner->content.children[split + index];
  leftInner->entryCount = split;
  uintptr_t leftFence = leftInner->getFenceKey();
  ;
  parent->updateSeparatorAfterSplit(rightFence, leftFence, rightInner);
  if (target <= leftFence) {
    inner = leftInner;
    rightInner->unlockExclusive();
  } else {
    inner = rightInner;
    leftInner->unlockExclusive();
  }
}

template <class Payload>
void BTreeLookup::Tree<Payload>::splitLeaf(Node *&leaf, Node *&parent, uintptr_t fence, uintptr_t target)
// Split a leaf node
{
  // Check for the root
  handleRootSplit(leaf, parent);

  // Create two leaf node
  uintptr_t rightFence = fence;
  Node *leftLeaf = leaf;
  Node *rightLeaf = allocateNode(false);
  unsigned split = leftLeaf->entryCount / 2;
  rightLeaf->entryCount = leftLeaf->entryCount - split;
  for (unsigned index = 0; index != rightLeaf->entryCount; ++index)
    rightLeaf->content.entries[index] = leftLeaf->content.entries[split + index];
  leftLeaf->entryCount = split;
  uintptr_t leftFence = rightLeaf->content.entries[0].base - 1;
  parent->updateSeparatorAfterSplit(rightFence, leftFence, rightLeaf);
  if (target <= leftFence) {
    leaf = leftLeaf;
    rightLeaf->unlockExclusive();
  } else {
    leaf = rightLeaf;
    leftLeaf->unlockExclusive();
  }
}

template <class Payload>
BTreeLookup::Node<Payload> *BTreeLookup::Tree<Payload>::mergeNode(unsigned childSlot, Node *parent, uintptr_t target)
// Merge (or balance) child nodes
{
  assert(parent->entryCount > 1);

  // Choose the emptiest neighbor and lock both. The target child is already locked
  unsigned leftSlot;
  Node *leftNode, *rightNode;
  if ((childSlot == 0) || (((childSlot + 1) < parent->entryCount) && (parent->content.children[childSlot + 1].child->entryCount < parent->content.children[childSlot - 1].child->entryCount))) {
    leftSlot = childSlot;
    leftNode = parent->content.children[leftSlot].child;
    rightNode = parent->content.children[leftSlot + 1].child;
    rightNode->lockExclusive();
  } else {
    leftSlot = childSlot - 1;
    leftNode = parent->content.children[leftSlot].child;
    rightNode = parent->content.children[leftSlot + 1].child;
    leftNode->lockExclusive();
  }

  // Can we merge both nodes into one node?
  unsigned totalCount = leftNode->entryCount + rightNode->entryCount;
  unsigned maxCount = leftNode->isInner() ? Node::maxFanoutInner : Node::maxFanoutLeaf;
  if (totalCount <= maxCount) {
    // Merge into the parent?
    if (parent->entryCount == 2) {
      // Merge children into parent. This can only happen at the root
      if (leftNode->isInner()) {
        for (unsigned index = 0; index != leftNode->entryCount; ++index)
          parent->content.children[index] = leftNode->content.children[index];
        for (unsigned index = 0; index != rightNode->entryCount; ++index)
          parent->content.children[index + leftNode->entryCount] = rightNode->content.children[index];
      } else {
        parent->type = Node::Leaf;
        for (unsigned index = 0; index != leftNode->entryCount; ++index)
          parent->content.entries[index] = leftNode->content.entries[index];
        for (unsigned index = 0; index != rightNode->entryCount; ++index)
          parent->content.entries[index + leftNode->entryCount] = rightNode->content.entries[index];
      }
      parent->entryCount = totalCount;
      releaseNode(leftNode);
      releaseNode(rightNode);
      return parent;
    } else {
      // Regular merge
      if (leftNode->isInner()) {
        for (unsigned index = 0; index != rightNode->entryCount; ++index)
          leftNode->content.children[leftNode->entryCount++] = rightNode->content.children[index];
      } else {
        for (unsigned index = 0; index != rightNode->entryCount; ++index)
          leftNode->content.entries[leftNode->entryCount++] = rightNode->content.entries[index];
      }
      parent->content.children[leftSlot].separator = parent->content.children[leftSlot + 1].separator;
      for (unsigned index = leftSlot + 1; index + 1 < parent->entryCount; ++index)
        parent->content.children[index] = parent->content.children[index + 1];
      parent->entryCount--;
      releaseNode(rightNode);
      parent->unlockExclusive();
      return leftNode;
    }
  }

  // No merge possible, rebalance instead
  if (leftNode->entryCount > rightNode->entryCount) {
    // Shift from left to right
    unsigned toShift = (leftNode->entryCount - rightNode->entryCount) / 2;
    if (leftNode->isInner()) {
      for (unsigned index = 0; index != rightNode->entryCount; ++index) {
        unsigned pos = rightNode->entryCount - 1 - index;
        rightNode->content.children[pos + toShift] = rightNode->content.children[pos];
      }
      for (unsigned index = 0; index != toShift; ++index)
        rightNode->content.children[index] = leftNode->content.children[leftNode->entryCount - toShift + index];
    } else {
      for (unsigned index = 0; index != rightNode->entryCount; ++index) {
        unsigned pos = rightNode->entryCount - 1 - index;
        rightNode->content.entries[pos + toShift] = rightNode->content.entries[pos];
      }
      for (unsigned index = 0; index != toShift; ++index)
        rightNode->content.entries[index] = leftNode->content.entries[leftNode->entryCount - toShift + index];
    }
    leftNode->entryCount -= toShift;
    rightNode->entryCount += toShift;
  } else {
    // Shift from right to left
    unsigned toShift = (rightNode->entryCount - leftNode->entryCount) / 2;
    if (leftNode->isInner()) {
      for (unsigned index = 0; index != toShift; ++index)
        leftNode->content.children[leftNode->entryCount + index] = rightNode->content.children[index];
      for (unsigned index = 0; index != rightNode->entryCount - toShift; ++index)
        rightNode->content.children[index] = rightNode->content.children[index + toShift];
    } else {
      for (unsigned index = 0; index != toShift; ++index)
        leftNode->content.entries[leftNode->entryCount + index] = rightNode->content.entries[index];
      for (unsigned index = 0; index != rightNode->entryCount - toShift; ++index)
        rightNode->content.entries[index] = rightNode->content.entries[index + toShift];
    }
    leftNode->entryCount += toShift;
    rightNode->entryCount -= toShift;
  }
  uintptr_t leftFence;
  if (leftNode->isLeaf()) {
    leftFence = rightNode->content.entries[0].base - 1;
  } else {
    leftFence = leftNode->getFenceKey();
  }
  parent->content.children[leftSlot].separator = leftFence;
  parent->unlockExclusive();
  if (target <= leftFence) {
    rightNode->unlockExclusive();
    return leftNode;
  } else {
    leftNode->unlockExclusive();
    return rightNode;
  }
}

template <class Payload>
bool BTreeLookup::Tree<Payload>::insert(uintptr_t base, uintptr_t size, Payload payload, Node **secondaryRoot)
// Insert an FDE
{
  // Sanity check
  if (!size)
    return false;

  // Access the root. Usually we simply take the root node, but when constructing a second tree in parallel
  // we use a secondary root instead
  Node *iter, *parent = nullptr;
  if (!secondaryRoot) {
    rootLock.lockExclusive();
    iter = root;
    if (iter) {
      iter->lockExclusive();
    } else {
      root = iter = allocateNode(false);
    }
    rootLock.unlockExclusive();
  } else {
    iter = *secondaryRoot;
    if (iter) {
      iter->lockExclusive();
    } else {
      *secondaryRoot = iter = allocateNode(false);
    }
  }

  // Walk down the btree with classic lock coupling and eager splits.
  // Strictly speaking this is not performance optimal, we could use
  // optimistic lock coupling until we hit a node that has to be modified.
  // But that is more difficult to implement and dlopen/dlclose acquires
  // a global lock anyway, we would not gain anything in concurrency here.

  uintptr_t fence = maxSeparator;
  while (iter->isInner()) {
    // Use eager splits to avoid lock coupling up
    if (iter->entryCount == Node::maxFanoutInner)
      splitInner(iter, parent, base);

    unsigned slot = iter->findInnerSlot(base);
    if (parent)
      parent->unlockExclusive();
    parent = iter;
    fence = iter->content.children[slot].separator;
    iter = iter->content.children[slot].child;
    iter->lockExclusive();
  }

  // Make sure we have space
  if (iter->entryCount == Node::maxFanoutLeaf)
    splitLeaf(iter, parent, fence, base);
  if (parent)
    parent->unlockExclusive();

  // Insert in page
  unsigned slot = iter->findLeafSlot(base);
  if ((slot < iter->entryCount) && (iter->content.entries[slot].base == base)) {
    // duplicate entry, this should never happen
    iter->unlockExclusive();
    return false;
  }
  for (unsigned index = iter->entryCount; index > slot; --index)
    iter->content.entries[index] = iter->content.entries[index - 1];
  auto &e = iter->content.entries[slot];
  e.base = base;
  e.size = size;
  e.payload = payload;
  iter->entryCount++;
  iter->unlockExclusive();
  return true;
}

template <class Payload>
bool BTreeLookup::Tree<Payload>::remove(uintptr_t base)
// Insert an FDE
{
  // Access the root
  rootLock.lockExclusive();
  Node *iter = root;
  if (iter)
    iter->lockExclusive();
  rootLock.unlockExclusive();
  if (!iter)
    return false;

  // Same strategy as with insert, walk down with lock coupling and
  // merge eagerly
  while (iter->isInner()) {
    unsigned slot = iter->findInnerSlot(base);
    Node *next = iter->content.children[slot].child;
    next->lockExclusive();
    if (next->needsMerge()) {
      // Use eager merges to avoid lock coupling up
      iter = mergeNode(slot, iter, base);
    } else {
      iter->unlockExclusive();
      iter = next;
    }
  }

  // Remove existing entry
  unsigned slot = iter->findLeafSlot(base);
  if ((slot >= iter->entryCount) || (iter->content.entries[slot].base != base)) {
    // not found, this should never happen
    iter->unlockExclusive();
    return false;
  }
  for (unsigned index = slot; index + 1 < iter->entryCount; ++index)
    iter->content.entries[index] = iter->content.entries[index + 1];
  iter->entryCount--;
  iter->unlockExclusive();
  return true;
}

template <class Payload>
bool BTreeLookup::Tree<Payload>::lookup(uintptr_t targetAddr, LookupResult &result) {
  // The unwinding tables are mostly static, they only change when shared libraries are
  // added or removed. This makes it extremely unlikely that they change during a given
  // unwinding sequence. Thus, we optimize for the contention free case and use optimistic
  // lock coupling. This does not require any writes to shared state, instead we validate
  // every read. It is important that we do not trust any value that we have read until
  // we call validate again. Data can change at arbitrary points in time, thus we always
  // copy something into a local variable and validate again before acting on the read.
  // In the unlikely event that we encounter a concurrent change we simply restart and try again.

restart:
  Node *iter;
  uintptr_t lock;
  {
    // Accessing the root node requires defending against concurrent pointer changes
    // Thus we couple rootLock -> lock on root node -> validate rootLock
    if (!rootLock.lockOptimistic(lock))
      goto restart;
    iter = root;
    if (!rootLock.validate(lock))
      goto restart;
    if (!iter)
      return false;
    uintptr_t childLock;
    if ((!iter->lockOptimistic(childLock)) || (!rootLock.validate(lock)))
      goto restart;
    lock = childLock;
  }

  // Now we can walk down towards the right leaf node
  while (true) {
    auto type = iter->type;
    unsigned entryCount = iter->entryCount;
    if (!iter->validate(lock))
      goto restart;
    if (!entryCount)
      return false;

    if (type == Node::Inner) {
      // We cannot call findInnerSlot here because we can only trust our validated entries
      unsigned slot = 0;
      while (((slot + 1) < entryCount) && (iter->content.children[slot].separator < targetAddr))
        ++slot;
      Node *child = iter->content.children[slot].child;
      if (!iter->validate(lock))
        goto restart;

      // The node content can change at any point in time, thus we must interleave parent and child checks
      uintptr_t childLock;
      if (!child->lockOptimistic(childLock))
        goto restart;
      if (!iter->validate(lock))
        goto restart; // make sure we still point to the correct page after acquiring the optimistic lock

      // Go down
      iter = child;
      lock = childLock;
    } else {
      // We cannot call findLeafSlot here because we can only trust our validated entries
      unsigned slot = 0;
      while (((slot + 1) < entryCount) && (iter->content.entries[slot].base + iter->content.entries[slot].size <= targetAddr))
        ++slot;
      typename Node::LeafEntry entry = iter->content.entries[slot];
      if (!iter->validate(lock))
        goto restart;

      // Check if we have a hit
      if ((entry.base <= targetAddr) && (targetAddr < entry.base + entry.size)) {
        result.base = entry.base;
        result.size = entry.size;
        result.payload = entry.payload;
        return true;
      }
      return false;
    }
  }
}

BTreeLookup::BTreeLookup() {
  // We set the initialization state to 1 recognize calls at startup
  // before the constructor was executed. This does not enable
  // the btree yet, that has to be done in the sync method
  __atomic_store_n(&initialization, 1, __ATOMIC_SEQ_CST);
}

BTreeLookup::~BTreeLookup() {
  // The trees are released implicitly here
}

void BTreeLookup::sync() {
  // Recognize calls that happen too early at program start
  if (!isConstructed())
    return;

  // Prevent concurrent syncs. This is not a loss of concurrency, as the dl_iterate_phdr call below
  // synchronizes implicitly anyway
  mutex.lock();

  // Unfortunately we currently have no way to learn about loaded and unloaded shared
  // libraries. Thus, we rebuild the full tree at every sync. Updating instead of rebuilding
  // would be highly desirable, but the glibc lacks the notification hooks that we need for
  // that. The only good thing is that the tree is usually small, as we have one entry per
  // section and not one per function.

  Node<EHTableInfo>* newRoot = nullptr;
  CallbackInfo callbackInfo{this, &newRoot};
  dl_iterate_phdr(&iterateCallback, &callbackInfo);

  // Replace the tree
  tableTree.rootLock.lockExclusive();
  auto oldRoot = tableTree.root;
  __atomic_store_n(&tableTree.root, newRoot, __ATOMIC_SEQ_CST);
  tableTree.rootLock.unlockExclusive();

  mutex.unlock();

  // We can release the old pages outside the mutex
  if (oldRoot)
    tableTree.releaseTreeRecursively(oldRoot);
}

int BTreeLookup::iterateCallback(struct dl_phdr_info *pinfo, size_t /*size*/, void *data) {
  if (pinfo->dlpi_phnum == 0)
    return 0;

  // Check if we have unwind information at all
  Elf_Addr image_base = calculateImageBase(pinfo);
  EHHeaderParser<LocalAddressSpace>::EHHeaderInfo hdrInfo;
  EHTableInfo ehInfo;
  bool found = false;
  for (Elf_Half i = pinfo->dlpi_phnum; i > 0; i--) {
    const Elf_Phdr *phdr = &pinfo->dlpi_phdr[i - 1];
    if (phdr->p_type == PT_GNU_EH_FRAME) {
      uintptr_t eh_frame_hdr_start = image_base + phdr->p_vaddr;
      ehInfo.dwarf_index_section = eh_frame_hdr_start;
      ehInfo.dwarf_index_section_length = phdr->p_memsz;
      if (EHHeaderParser<LocalAddressSpace>::decodeEHHdr(LocalAddressSpace::sThisAddressSpace, eh_frame_hdr_start, phdr->p_memsz, hdrInfo)) {
        // .eh_frame_hdr records the start of .eh_frame, but not its size.
        // Rely on a zero terminator to find the end of the section.
        ehInfo.dwarf_section = hdrInfo.eh_frame_ptr;
        ehInfo.dwarf_section_length = SIZE_MAX;
        found = true;
        break;
      }
    }
  }
  if (!found)
    return 0;

  // If yes, register each code section with the unwinding information
  CallbackInfo info = *static_cast<CallbackInfo *>(data);
  for (Elf_Half i = pinfo->dlpi_phnum; i > 0; i--) {
    const Elf_Phdr *phdr = &pinfo->dlpi_phdr[i - 1];
    if (phdr->p_type == PT_LOAD) {
      uintptr_t dso_base = image_base + phdr->p_vaddr;
      uintptr_t text_segment_length = phdr->p_memsz;
      info.lookup->tableTree.insert(dso_base, text_segment_length, ehInfo, info.newRoot);
    }
  }
  return 0;
}

bool BTreeLookup::insertFrame(uintptr_t base, uintptr_t size, uintptr_t fde)
// Explicitly register a frame
{
  if (!isConstructed())
    return false;

  // And insert the frame
  ExplicitFrameInfo info;
  info.fde = fde;
  return frameTree.insert(base, size, info);
}

bool BTreeLookup::removeFrame(uintptr_t base)
// Remove a previously registered frame
{
  if (!isConstructed())
    return false;

  return frameTree.remove(base);
}

uintptr_t BTreeLookup::findFDE(uintptr_t targetAddr)
// Find the corresponding unwinding info for the given address
{
  // Check the regular unwinding tables first
  Tree<EHTableInfo>::LookupResult tableResult;
  if (tableTree.lookup(targetAddr, tableResult)) {
    CFI_Parser<LocalAddressSpace>::FDE_Info fdeInfo;
    CFI_Parser<LocalAddressSpace>::CIE_Info cieInfo;
    bool foundFDE = false;
    auto& _addressSpace = LocalAddressSpace::sThisAddressSpace;
    if (!foundFDE && (tableResult.payload.dwarf_index_section != 0))
      foundFDE = EHHeaderParser<LocalAddressSpace>::findFDE(_addressSpace, targetAddr, tableResult.payload.dwarf_index_section, (uint32_t)tableResult.payload.dwarf_index_section_length, &fdeInfo, &cieInfo);
    if (!foundFDE) {
      // Still not found, do full scan of __eh_frame section.
      foundFDE = CFI_Parser<LocalAddressSpace>::findFDE(_addressSpace, targetAddr, tableResult.payload.dwarf_section, tableResult.payload.dwarf_section_length, 0, &fdeInfo, &cieInfo);
    }
    if (foundFDE)
       return fdeInfo.fdeStart;
  }

  // Not found, check if it is an explicit frame
  Tree<ExplicitFrameInfo>::LookupResult frameResult;
  if (frameTree.lookup(targetAddr, frameResult)) {
    return frameResult.payload.fde;
  }

  return 0;
}

}

#endif // __BTREE_LOOKUP_HPP__
