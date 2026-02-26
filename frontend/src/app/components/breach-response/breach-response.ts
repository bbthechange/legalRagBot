import { Component, inject, signal, computed } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatePipe } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { BreachResponse } from '../../models/api.models';
import { ActionBarComponent } from '../action-bar/action-bar';

@Component({
  selector: 'app-breach-response',
  imports: [FormsModule, DatePipe, ActionBarComponent],
  templateUrl: './breach-response.html',
  styleUrl: './breach-response.scss',
})
export class BreachResponseComponent {
  private readonly api = inject(ApiService);

  loading = signal(false);
  error = signal<string | null>(null);
  response = signal<BreachResponse | null>(null);
  viewMode = signal<'timeline' | 'matrix'>('timeline');

  dataTypes = [
    { id: 'ssn', label: 'SSN' },
    { id: 'financial', label: 'Financial Account' },
    { id: 'drivers_license', label: "Driver's License" },
    { id: 'email_password', label: 'Email + Password' },
    { id: 'medical', label: 'Medical Records' },
    { id: 'biometric', label: 'Biometric' },
  ];

  states = ['CA', 'NY', 'TX', 'FL', 'IL', 'WA', 'MA', 'CO', 'VA', 'CT'];

  encryptionOptions = [
    { value: 'unencrypted', label: 'Unencrypted' },
    { value: 'encrypted', label: 'Encrypted' },
    { value: 'partial', label: 'Partially Encrypted' },
    { value: 'unknown', label: 'Unknown' },
  ];

  entityTypes = [
    { value: 'for_profit', label: 'For-Profit' },
    { value: 'healthcare', label: 'Healthcare' },
    { value: 'financial_institution', label: 'Financial Institution' },
    { value: 'government', label: 'Government' },
  ];

  selectedDataTypes = signal<Set<string>>(new Set());
  selectedStates = signal<Set<string>>(new Set());
  affectedCount = '';
  encryptionStatus = 'unknown';
  entityType = 'for_profit';
  discoveryDate = '';

  summary = computed(() => this.response()?.summary);

  timelineEntries = computed(() => {
    const r = this.response();
    if (!r) return [];
    return [...r.state_analyses]
      .filter((s: any) => s.notification_required)
      .sort((a: any, b: any) => {
        if (!a.deadline) return 1;
        if (!b.deadline) return -1;
        return a.deadline.localeCompare(b.deadline);
      });
  });

  toggleDataType(id: string): void {
    const current = new Set(this.selectedDataTypes());
    if (current.has(id)) current.delete(id);
    else current.add(id);
    this.selectedDataTypes.set(current);
  }

  toggleState(state: string): void {
    const current = new Set(this.selectedStates());
    if (current.has(state)) current.delete(state);
    else current.add(state);
    this.selectedStates.set(current);
  }

  analyze(): void {
    if (this.selectedDataTypes().size === 0 || this.selectedStates().size === 0 || this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    this.response.set(null);

    const count = this.affectedCount ? Number(this.affectedCount) : 'unknown';

    this.api.analyzeBreach({
      data_types_compromised: Array.from(this.selectedDataTypes()),
      affected_states: Array.from(this.selectedStates()),
      number_of_affected_individuals: Number.isFinite(count as number) ? count : 'unknown',
      encryption_status: this.encryptionStatus,
      entity_type: this.entityType,
      date_of_discovery: this.discoveryDate || null,
    }).subscribe({
      next: (res) => {
        this.response.set(res);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(err?.error?.detail || err?.message || 'Breach analysis failed. Check that the backend is running.');
        this.loading.set(false);
      },
    });
  }
}
