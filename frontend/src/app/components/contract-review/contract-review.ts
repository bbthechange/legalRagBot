import { Component, inject, signal, computed } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { ContractReviewResponse } from '../../models/api.models';
import { ActionBarComponent } from '../action-bar/action-bar';
import { RiskBadgeComponent } from '../risk-badge/risk-badge';

@Component({
  selector: 'app-contract-review',
  imports: [FormsModule, ActionBarComponent, RiskBadgeComponent],
  templateUrl: './contract-review.html',
  styleUrl: './contract-review.scss',
})
export class ContractReviewComponent {
  private readonly api = inject(ApiService);

  contractText = '';
  selectedPlaybook = signal('saas-vendor-review');
  loading = signal(false);
  error = signal<string | null>(null);
  response = signal<ContractReviewResponse | null>(null);
  expandedClause = signal<number | null>(null);
  sortByRisk = signal(false);
  inputExpanded = signal(true);

  playbooks = [
    { id: 'saas-vendor-review', label: 'SaaS Vendor Review', desc: 'Standard review for SaaS vendor agreements' },
    { id: 'nda-review', label: 'NDA Review', desc: 'Non-disclosure agreement review against firm positions' },
  ];

  playbookLabel = computed(() =>
    this.playbooks.find(p => p.id === this.selectedPlaybook())?.label ?? this.selectedPlaybook()
  );

  summary = computed(() => this.response()?.summary);

  clauses = computed(() => {
    const r = this.response();
    if (!r) return [];
    const list = [...r.clause_analyses];
    if (this.sortByRisk()) {
      const order: Record<string, number> = { walk_away: 0, high: 1, medium: 2, fallback: 3, low: 4, preferred: 5, not_covered: 6 };
      list.sort((a: any, b: any) => (order[a.risk_level] ?? 9) - (order[b.risk_level] ?? 9));
    }
    return list;
  });

  walkAways = computed(() => {
    const r = this.response();
    if (!r) return [];
    return r.clause_analyses.filter((c: any) => c.playbook_match === 'walk_away' || c.risk_level?.toLowerCase() === 'high');
  });

  analyze(): void {
    if (!this.contractText.trim() || this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    this.response.set(null);

    this.api.reviewContract({
      contract_text: this.contractText,
      playbook: this.selectedPlaybook(),
    }).subscribe({
      next: (res) => {
        this.response.set(res);
        this.loading.set(false);
        this.inputExpanded.set(false);
      },
      error: (err) => {
        this.error.set(err?.error?.detail || err?.message || 'Contract review failed. Check that the backend is running.');
        this.loading.set(false);
      },
    });
  }

  toggleClause(index: number): void {
    this.expandedClause.set(this.expandedClause() === index ? null : index);
  }

  matchLabel(match: string): string {
    const map: Record<string, string> = {
      preferred: 'Preferred',
      fallback: 'Fallback',
      walk_away: 'Walk-Away',
      not_covered: 'Not Covered',
    };
    return map[match] || match;
  }
}
