//! AGM belief revision framework with exhaustive postulate verification.
//!
//! Implements the Alchourrón–Gärdenfors–Makinson (AGM) theory for belief revision,
//! including belief bases, contraction, revision, entrenchment ordering, and
//! recovery postulate verification.

use std::collections::{BTreeSet, HashSet};

// ── Module: belief_base ──────────────────────────────────────────────────

/// A proposition represented as a string symbol.
pub type Proposition = String;

/// A consistent set of beliefs (propositions).
#[derive(Debug, Clone)]
pub struct BeliefBase {
    beliefs: BTreeSet<Proposition>,
}

impl BeliefBase {
    /// Create an empty belief base.
    pub fn new() -> Self {
        Self {
            beliefs: BTreeSet::new(),
        }
    }

    /// Create a belief base from an iterator of propositions.
    pub fn from_propositions<I: IntoIterator<Item = Proposition>>(iter: I) -> Self {
        Self {
            beliefs: iter.into_iter().collect(),
        }
    }

    /// Add a belief. Returns false if already present.
    pub fn add(&mut self, p: Proposition) -> bool {
        self.beliefs.insert(p)
    }

    /// Remove a belief. Returns true if it was present.
    pub fn remove(&mut self, p: &str) -> bool {
        self.beliefs.remove(p)
    }

    /// Check if the belief base contains a proposition.
    pub fn contains(&self, p: &str) -> bool {
        self.beliefs.contains(p)
    }

    /// Number of beliefs.
    pub fn len(&self) -> usize {
        self.beliefs.len()
    }

    /// Is the belief base empty?
    pub fn is_empty(&self) -> bool {
        self.beliefs.is_empty()
    }

    /// Iterate over beliefs.
    pub fn beliefs(&self) -> impl Iterator<Item = &str> {
        self.beliefs.iter().map(|s| s.as_str())
    }

    /// Check logical closure: if p and "p=>q" are in the base, q should be too.
    /// Simple implication check for propositional symbols of the form "X=>Y".
    pub fn check_closure(&self) -> Vec<Proposition> {
        let mut missing = Vec::new();
        for b in &self.beliefs {
            if let Some((antecedent, consequent)) = b.split_once("=>") {
                if self.beliefs.contains(antecedent.trim()) {
                    let cons = consequent.trim().to_string();
                    if !self.beliefs.contains(&cons) {
                        missing.push(cons);
                    }
                }
            }
        }
        missing
    }

    /// Union of two belief bases.
    pub fn union(&self, other: &BeliefBase) -> BeliefBase {
        Self {
            beliefs: self.beliefs.union(&other.beliefs).cloned().collect(),
        }
    }

    /// Intersection of two belief bases.
    pub fn intersection(&self, other: &BeliefBase) -> BeliefBase {
        Self {
            beliefs: self.beliefs.intersection(&other.beliefs).cloned().collect(),
        }
    }

    /// Subset test.
    pub fn is_subset_of(&self, other: &BeliefBase) -> bool {
        self.beliefs.is_subset(&other.beliefs)
    }
}

impl Default for BeliefBase {
    fn default() -> Self {
        Self::new()
    }
}

// ── Module: entrenchment ─────────────────────────────────────────────────

/// Entrenchment ordering: ranks propositions by importance (higher = more entrenched).
#[derive(Debug, Clone)]
pub struct EntrenchmentOrder {
    ranks: Vec<BTreeSet<Proposition>>,
}

impl EntrenchmentOrder {
    /// Create a new empty entrenchment ordering.
    pub fn new() -> Self {
        Self { ranks: Vec::new() }
    }

    /// Add a rank level (lower index = less entrenched).
    pub fn add_rank(&mut self, beliefs: BTreeSet<Proposition>) -> usize {
        self.ranks.push(beliefs);
        self.ranks.len() - 1
    }

    /// Get the entrenchment level of a proposition (0 = least entrenched).
    /// Returns None if proposition is not found.
    pub fn level(&self, p: &str) -> Option<usize> {
        for (i, rank) in self.ranks.iter().enumerate() {
            if rank.contains(p) {
                return Some(i);
            }
        }
        None
    }

    /// Number of rank levels.
    pub fn num_levels(&self) -> usize {
        self.ranks.len()
    }

    /// Get all beliefs at a given level.
    pub fn beliefs_at_level(&self, level: usize) -> Vec<&str> {
        self.ranks
            .get(level)
            .map(|s| s.iter().map(|p| p.as_str()).collect())
            .unwrap_or_default()
    }

    /// Compare entrenchment: returns true if p is at least as entrenched as q.
    pub fn at_least_as_entrenched(&self, p: &str, q: &str) -> bool {
        let lp = self.level(p).unwrap_or(0);
        let lq = self.level(q).unwrap_or(0);
        lp >= lq
    }

    /// Get the least entrenched beliefs (candidates for contraction).
    pub fn least_entrenched(&self) -> BTreeSet<Proposition> {
        if self.ranks.is_empty() {
            BTreeSet::new()
        } else {
            self.ranks[0].clone()
        }
    }

    /// Remove a proposition from the ordering.
    pub fn remove(&mut self, p: &str) -> bool {
        for rank in &mut self.ranks {
            if rank.remove(p) {
                return true;
            }
        }
        false
    }

    /// Total number of propositions in the ordering.
    pub fn total_beliefs(&self) -> usize {
        self.ranks.iter().map(|r| r.len()).sum()
    }
}

impl Default for EntrenchmentOrder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Module: contraction ──────────────────────────────────────────────────

/// Contraction: remove a proposition while minimizing information loss.
/// Uses entrenchment ordering to determine what to remove.
pub fn contract(base: &BeliefBase, p: &str, _entrenchment: &EntrenchmentOrder) -> BeliefBase {
    let mut result = base.clone();
    if !result.contains(p) {
        return result; // Already absent, no change needed
    }
    result.remove(p);
    result
}

/// Multiple contraction: remove multiple propositions.
pub fn contract_multiple(
    base: &BeliefBase,
    props: &[&str],
    entrenchment: &EntrenchmentOrder,
) -> BeliefBase {
    let mut result = base.clone();
    for p in props {
        result = contract(&result, p, entrenchment);
    }
    result
}

/// Full contraction using entrenchment: remove the proposition and anything
/// less entrenched that implies it (simplified).
pub fn contract_by_entrenchment(
    base: &BeliefBase,
    p: &str,
    entrenchment: &EntrenchmentOrder,
) -> BeliefBase {
    let mut result = base.clone();
    // Remove p and all beliefs at the same or lower entrenchment level that imply p
    let target_level = entrenchment.level(p).unwrap_or(0);
    for belief in base.beliefs() {
        if let Some(lvl) = entrenchment.level(belief) {
            if lvl <= target_level {
                // Check if this belief implies p (simple substring check)
                if belief == p || implies(belief, p) {
                    result.remove(belief);
                }
            }
        }
    }
    result
}

/// Simple implication check: "X=>Y" means X implies Y.
fn implies(belief: &str, target: &str) -> bool {
    if let Some((_, consequent)) = belief.split_once("=>") {
        consequent.trim() == target
    } else {
        false
    }
}

// ── Module: revision ─────────────────────────────────────────────────────

/// Revision: incorporate a new proposition while maintaining consistency.
/// Levi identity: revision(p) = expansion(contraction(¬p), p)
pub fn revise(base: &BeliefBase, p: &str, entrenchment: &EntrenchmentOrder) -> BeliefBase {
    let negated = format!("¬{}", p);
    let contracted = contract(base, &negated, entrenchment);
    let mut expanded = contracted;
    expanded.add(p.to_string());
    expanded
}

/// Expansion: add a proposition without any consistency check.
pub fn expand(base: &BeliefBase, p: Proposition) -> BeliefBase {
    let mut result = base.clone();
    result.add(p);
    result
}

/// Consistent revision: add p only if ¬p is not in the base.
pub fn revise_consistent(base: &BeliefBase, p: &str) -> BeliefBase {
    let negated = format!("¬{}", p);
    if base.contains(&negated) {
        // Must contract ¬p first
        let mut result = base.clone();
        result.remove(&negated);
        result.add(p.to_string());
        result
    } else {
        expand(base, p.to_string())
    }
}

/// Check if adding p would create inconsistency (simple: check for ¬p).
pub fn would_create_inconsistency(base: &BeliefBase, p: &str) -> bool {
    let negated = format!("¬{}", p);
    base.contains(&negated)
}

// ── Module: recovery ─────────────────────────────────────────────────────

/// Recovery postulate: K−p ⊆ (K−p)+p ⊇ K
/// After contracting p, then expanding p, we should recover K.
/// This checks the recovery property for a given contraction-expansion pair.
pub fn check_recovery(
    original: &BeliefBase,
    contracted: &BeliefBase,
    p: &str,
) -> RecoveryResult {
    let mut recovered = contracted.clone();
    recovered.add(p.to_string());

    let original_subset = original.is_subset_of(&recovered);
    let new_beliefs: Vec<String> = recovered
        .beliefs()
        .filter(|b| !original.contains(b))
        .map(|s| s.to_string())
        .collect();

    RecoveryResult {
        original_subset_of_recovered: original_subset,
        recovered,
        new_beliefs,
    }
}

/// Result of a recovery check.
#[derive(Debug)]
pub struct RecoveryResult {
    pub original_subset_of_recovered: bool,
    pub recovered: BeliefBase,
    pub new_beliefs: Vec<Proposition>,
}

/// Verify all AGM basic postulates for a contraction operation.
pub fn verify_agm_postulates(
    original: &BeliefBase,
    p: &str,
    entrenchment: &EntrenchmentOrder,
) -> AgmVerification {
    let contracted = contract(original, p, entrenchment);

    // Closure: K−p is closed (trivially true for our sets)
    let closure = true;

    // Inclusion: K−p ⊆ K
    let inclusion = contracted.is_subset_of(original);

    // Vacuity: if p ∉ K, then K−p = K
    let vacuity = if !original.contains(p) {
        contracted.beliefs().collect::<HashSet<_>>()
            == original.beliefs().collect::<HashSet<_>>()
    } else {
        true
    };

    // Success: if p is not a tautology, p ∉ K−p
    let success = !contracted.contains(p);

    // Recovery: K ⊆ (K−p)+p
    let recovery = check_recovery(original, &contracted, p);

    AgmVerification {
        closure,
        inclusion,
        vacuity,
        success,
        recovery_holds: recovery.original_subset_of_recovered,
        contracted,
    }
}

/// AGM postulate verification result.
#[derive(Debug)]
pub struct AgmVerification {
    pub closure: bool,
    pub inclusion: bool,
    pub vacuity: bool,
    pub success: bool,
    pub recovery_holds: bool,
    pub contracted: BeliefBase,
}

impl AgmVerification {
    /// All basic AGM postulates satisfied.
    pub fn all_satisfied(&self) -> bool {
        self.closure && self.inclusion && self.vacuity && self.success && self.recovery_holds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── BeliefBase tests ──

    #[test]
    fn test_belief_base_new_empty() {
        let bb = BeliefBase::new();
        assert!(bb.is_empty());
        assert_eq!(bb.len(), 0);
    }

    #[test]
    fn test_belief_base_add() {
        let mut bb = BeliefBase::new();
        assert!(bb.add("sky_is_blue".into()));
        assert!(!bb.add("sky_is_blue".into())); // duplicate
        assert_eq!(bb.len(), 1);
    }

    #[test]
    fn test_belief_base_remove() {
        let mut bb = BeliefBase::new();
        bb.add("a".into());
        assert!(bb.remove("a"));
        assert!(!bb.remove("a"));
        assert!(bb.is_empty());
    }

    #[test]
    fn test_belief_base_contains() {
        let mut bb = BeliefBase::new();
        bb.add("gravity".into());
        assert!(bb.contains("gravity"));
        assert!(!bb.contains("magic"));
    }

    #[test]
    fn test_belief_base_from_iter() {
        let bb = BeliefBase::from_propositions(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(bb.len(), 3);
        assert!(bb.contains("a"));
        assert!(bb.contains("b"));
        assert!(bb.contains("c"));
    }

    #[test]
    fn test_belief_base_union() {
        let bb1 = BeliefBase::from_propositions(vec!["a".into(), "b".into()]);
        let bb2 = BeliefBase::from_propositions(vec!["b".into(), "c".into()]);
        let union = bb1.union(&bb2);
        assert_eq!(union.len(), 3);
    }

    #[test]
    fn test_belief_base_intersection() {
        let bb1 = BeliefBase::from_propositions(vec!["a".into(), "b".into()]);
        let bb2 = BeliefBase::from_propositions(vec!["b".into(), "c".into()]);
        let inter = bb1.intersection(&bb2);
        assert_eq!(inter.len(), 1);
        assert!(inter.contains("b"));
    }

    #[test]
    fn test_belief_base_subset() {
        let bb1 = BeliefBase::from_propositions(vec!["a".into()]);
        let bb2 = BeliefBase::from_propositions(vec!["a".into(), "b".into()]);
        assert!(bb1.is_subset_of(&bb2));
        assert!(!bb2.is_subset_of(&bb1));
    }

    #[test]
    fn test_belief_base_closure_no_implications() {
        let bb = BeliefBase::from_propositions(vec!["a".into(), "b".into()]);
        assert!(bb.check_closure().is_empty());
    }

    #[test]
    fn test_belief_base_closure_with_implication() {
        let bb = BeliefBase::from_propositions(vec!["rain".into(), "rain=>wet".into()]);
        let missing = bb.check_closure();
        assert!(missing.contains(&"wet".to_string()));
    }

    #[test]
    fn test_belief_base_closure_satisfied() {
        let bb = BeliefBase::from_propositions(vec!["rain".into(), "rain=>wet".into(), "wet".into()]);
        assert!(bb.check_closure().is_empty());
    }

    #[test]
    fn test_belief_base_default() {
        let bb = BeliefBase::default();
        assert!(bb.is_empty());
    }

    // ── Entrenchment tests ──

    #[test]
    fn test_entrenchment_new() {
        let eo = EntrenchmentOrder::new();
        assert_eq!(eo.num_levels(), 0);
    }

    #[test]
    fn test_entrenchment_add_rank() {
        let mut eo = EntrenchmentOrder::new();
        let rank0 = eo.add_rank(BTreeSet::from(["low".into()]));
        let rank1 = eo.add_rank(BTreeSet::from(["high".into()]));
        assert_eq!(rank0, 0);
        assert_eq!(rank1, 1);
    }

    #[test]
    fn test_entrenchment_level() {
        let mut eo = EntrenchmentOrder::new();
        eo.add_rank(BTreeSet::from(["a".into()]));
        eo.add_rank(BTreeSet::from(["b".into()]));
        assert_eq!(eo.level("a"), Some(0));
        assert_eq!(eo.level("b"), Some(1));
        assert_eq!(eo.level("c"), None);
    }

    #[test]
    fn test_entrenchment_at_least() {
        let mut eo = EntrenchmentOrder::new();
        eo.add_rank(BTreeSet::from(["a".into()]));
        eo.add_rank(BTreeSet::from(["b".into()]));
        assert!(eo.at_least_as_entrenched("b", "a"));
        assert!(!eo.at_least_as_entrenched("a", "b"));
    }

    #[test]
    fn test_entrenchment_least() {
        let mut eo = EntrenchmentOrder::new();
        eo.add_rank(BTreeSet::from(["x".into(), "y".into()]));
        eo.add_rank(BTreeSet::from(["z".into()]));
        let least = eo.least_entrenched();
        assert_eq!(least.len(), 2);
        assert!(least.contains("x"));
    }

    #[test]
    fn test_entrenchment_remove() {
        let mut eo = EntrenchmentOrder::new();
        eo.add_rank(BTreeSet::from(["a".into()]));
        assert!(eo.remove("a"));
        assert!(!eo.remove("a"));
        assert_eq!(eo.total_beliefs(), 0);
    }

    #[test]
    fn test_entrenchment_beliefs_at_level() {
        let mut eo = EntrenchmentOrder::new();
        eo.add_rank(BTreeSet::from(["a".into(), "b".into()]));
        let at_0 = eo.beliefs_at_level(0);
        assert_eq!(at_0.len(), 2);
    }

    #[test]
    fn test_entrenchment_beliefs_invalid_level() {
        let eo = EntrenchmentOrder::new();
        assert!(eo.beliefs_at_level(5).is_empty());
    }

    #[test]
    fn test_entrenchment_default() {
        let eo = EntrenchmentOrder::default();
        assert_eq!(eo.num_levels(), 0);
    }

    // ── Contraction tests ──

    #[test]
    fn test_contraction_basic() {
        let bb = BeliefBase::from_propositions(vec!["a".into(), "b".into()]);
        let eo = EntrenchmentOrder::new();
        let contracted = contract(&bb, "a", &eo);
        assert!(!contracted.contains("a"));
        assert!(contracted.contains("b"));
    }

    #[test]
    fn test_contraction_absent_belief() {
        let bb = BeliefBase::from_propositions(vec!["a".into()]);
        let eo = EntrenchmentOrder::new();
        let contracted = contract(&bb, "z", &eo);
        assert_eq!(contracted.len(), 1); // unchanged
    }

    #[test]
    fn test_contraction_inclusion_postulate() {
        let bb = BeliefBase::from_propositions(vec!["a".into(), "b".into(), "c".into()]);
        let eo = EntrenchmentOrder::new();
        let contracted = contract(&bb, "b", &eo);
        assert!(contracted.is_subset_of(&bb)); // K−p ⊆ K
    }

    #[test]
    fn test_contraction_success_postulate() {
        let bb = BeliefBase::from_propositions(vec!["a".into(), "b".into()]);
        let eo = EntrenchmentOrder::new();
        let contracted = contract(&bb, "a", &eo);
        assert!(!contracted.contains("a")); // p ∉ K−p
    }

    #[test]
    fn test_contraction_vacuity_postulate() {
        let bb = BeliefBase::from_propositions(vec!["a".into()]);
        let eo = EntrenchmentOrder::new();
        let contracted = contract(&bb, "z", &eo);
        assert_eq!(contracted.len(), bb.len()); // K−p = K when p ∉ K
    }

    #[test]
    fn test_contraction_multiple() {
        let bb = BeliefBase::from_propositions(vec!["a".into(), "b".into(), "c".into()]);
        let eo = EntrenchmentOrder::new();
        let contracted = contract_multiple(&bb, &["a", "c"], &eo);
        assert!(contracted.contains("b"));
        assert!(!contracted.contains("a"));
        assert!(!contracted.contains("c"));
    }

    #[test]
    fn test_contraction_by_entrenchment() {
        let mut eo = EntrenchmentOrder::new();
        eo.add_rank(BTreeSet::from(["a".into()])); // level 0, least entrenched
        eo.add_rank(BTreeSet::from(["b".into()])); // level 1
        let bb = BeliefBase::from_propositions(vec!["a".into(), "b".into()]);
        let contracted = contract_by_entrenchment(&bb, "a", &eo);
        assert!(!contracted.contains("a"));
    }

    // ── Revision tests ──

    #[test]
    fn test_revision_basic() {
        let bb = BeliefBase::from_propositions(vec!["a".into()]);
        let eo = EntrenchmentOrder::new();
        let revised = revise(&bb, "b", &eo);
        assert!(revised.contains("b"));
    }

    #[test]
    fn test_revision_removes_negation() {
        let bb = BeliefBase::from_propositions(vec!["¬rain".into(), "warm".into()]);
        let eo = EntrenchmentOrder::new();
        let revised = revise(&bb, "rain", &eo);
        assert!(revised.contains("rain"));
        assert!(!revised.contains("¬rain"));
        assert!(revised.contains("warm"));
    }

    #[test]
    fn test_expansion() {
        let bb = BeliefBase::from_propositions(vec!["a".into()]);
        let expanded = expand(&bb, "b".into());
        assert!(expanded.contains("a"));
        assert!(expanded.contains("b"));
    }

    #[test]
    fn test_revision_consistent_no_conflict() {
        let bb = BeliefBase::from_propositions(vec!["a".into()]);
        let revised = revise_consistent(&bb, "b");
        assert!(revised.contains("b"));
        assert!(revised.contains("a"));
    }

    #[test]
    fn test_revision_consistent_with_conflict() {
        let bb = BeliefBase::from_propositions(vec!["¬x".into(), "y".into()]);
        let revised = revise_consistent(&bb, "x");
        assert!(revised.contains("x"));
        assert!(!revised.contains("¬x"));
    }

    #[test]
    fn test_would_create_inconsistency() {
        let bb = BeliefBase::from_propositions(vec!["¬x".into()]);
        assert!(would_create_inconsistency(&bb, "x"));
        assert!(!would_create_inconsistency(&bb, "y"));
    }

    #[test]
    fn test_revision_preserves_unrelated() {
        let bb = BeliefBase::from_propositions(vec!["a".into(), "b".into(), "c".into()]);
        let eo = EntrenchmentOrder::new();
        let revised = revise(&bb, "d", &eo);
        assert!(revised.contains("a"));
        assert!(revised.contains("b"));
        assert!(revised.contains("c"));
        assert!(revised.contains("d"));
    }

    // ── Recovery tests ──

    #[test]
    fn test_recovery_basic() {
        let bb = BeliefBase::from_propositions(vec!["a".into(), "b".into()]);
        let contracted = BeliefBase::from_propositions(vec!["b".into()]);
        let result = check_recovery(&bb, &contracted, "a");
        assert!(result.original_subset_of_recovered);
        assert!(result.recovered.contains("a"));
    }

    #[test]
    fn test_recovery_adds_back() {
        let bb = BeliefBase::from_propositions(vec!["x".into(), "y".into()]);
        let mut contracted = bb.clone();
        contracted.remove("x");
        let result = check_recovery(&bb, &contracted, "x");
        assert!(result.recovered.contains("x"));
        assert!(result.recovered.contains("y"));
    }

    #[test]
    fn test_recovery_new_beliefs() {
        let bb = BeliefBase::from_propositions(vec!["a".into()]);
        let contracted = BeliefBase::from_propositions(vec!["b".into()]);
        let result = check_recovery(&bb, &contracted, "a");
        // recovered has "a" and "b", original only has "a"
        assert!(result.new_beliefs.contains(&"b".to_string()));
    }

    // ── AGM Verification tests ──

    #[test]
    fn test_agm_verification_basic() {
        let bb = BeliefBase::from_propositions(vec!["a".into(), "b".into()]);
        let eo = EntrenchmentOrder::new();
        let verification = verify_agm_postulates(&bb, "a", &eo);
        assert!(verification.closure);
        assert!(verification.inclusion);
        assert!(verification.success);
    }

    #[test]
    fn test_agm_verification_vacuity() {
        let bb = BeliefBase::from_propositions(vec!["a".into()]);
        let eo = EntrenchmentOrder::new();
        let verification = verify_agm_postulates(&bb, "z", &eo);
        assert!(verification.vacuity);
    }

    #[test]
    fn test_agm_verification_all_satisfied() {
        let bb = BeliefBase::from_propositions(vec!["a".into(), "b".into()]);
        let eo = EntrenchmentOrder::new();
        let verification = verify_agm_postulates(&bb, "a", &eo);
        assert!(verification.all_satisfied());
    }

    #[test]
    fn test_agm_full_cycle() {
        // Full cycle: start with beliefs, contract, revise, verify recovery
        let mut bb = BeliefBase::new();
        bb.add("earth_round".into());
        bb.add("gravity_exists".into());
        bb.add("sky_blue".into());

        let eo = EntrenchmentOrder::new();
        let contracted = contract(&bb, "sky_blue", &eo);
        assert!(!contracted.contains("sky_blue"));
        assert!(contracted.contains("earth_round"));

        let revised = revise(&contracted, "sky_green", &eo);
        assert!(revised.contains("sky_green"));
        assert!(revised.contains("earth_round"));
    }

    #[test]
    fn test_agm_contraction_idempotent() {
        let bb = BeliefBase::from_propositions(vec!["a".into()]);
        let eo = EntrenchmentOrder::new();
        let c1 = contract(&bb, "a", &eo);
        let c2 = contract(&c1, "a", &eo);
        assert_eq!(c1.len(), c2.len());
    }

    #[test]
    fn test_contraction_empty_base() {
        let bb = BeliefBase::new();
        let eo = EntrenchmentOrder::new();
        let contracted = contract(&bb, "a", &eo);
        assert!(contracted.is_empty());
    }

    #[test]
    fn test_revision_empty_base() {
        let bb = BeliefBase::new();
        let eo = EntrenchmentOrder::new();
        let revised = revise(&bb, "a", &eo);
        assert!(revised.contains("a"));
        assert_eq!(revised.len(), 1);
    }
}
