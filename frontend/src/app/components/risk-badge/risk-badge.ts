import { Component, input } from '@angular/core';

@Component({
  selector: 'app-risk-badge',
  template: `
    <span class="risk-badge" [class]="'risk-badge risk-' + normalizedLevel()">
      {{ label() || normalizedLevel().toUpperCase() + ' RISK' }}
    </span>
  `,
  styles: [`
    .risk-badge {
      display: inline-flex;
      align-items: center;
      padding: 3px 10px;
      font-family: var(--font-body);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      border-radius: 2px;
      white-space: nowrap;
    }
    .risk-high {
      background: var(--risk-high);
      color: #fff;
    }
    .risk-medium {
      background: var(--risk-medium);
      color: #fff;
    }
    .risk-low {
      background: var(--risk-low);
      color: #fff;
    }
    .risk-gray {
      background: var(--risk-gray);
      color: #fff;
    }
    .risk-preferred {
      background: var(--risk-low);
      color: #fff;
    }
    .risk-fallback {
      background: var(--risk-medium);
      color: #fff;
    }
    .risk-walk_away {
      background: var(--risk-high);
      color: #fff;
    }
    .risk-not_covered {
      background: var(--risk-gray);
      color: #fff;
    }
  `],
})
export class RiskBadgeComponent {
  level = input<string>('');
  label = input<string>('');

  normalizedLevel(): string {
    const l = (this.level() || '').toLowerCase().replace(/[\s-]+/g, '_');
    if (l.includes('high') || l.includes('walk')) return l.includes('walk') ? 'walk_away' : 'high';
    if (l.includes('medium') || l.includes('moderate') || l.includes('fallback')) return l.includes('fallback') ? 'fallback' : 'medium';
    if (l.includes('low') || l.includes('preferred')) return l.includes('preferred') ? 'preferred' : 'low';
    if (l.includes('not_covered') || l.includes('not covered')) return 'not_covered';
    return 'gray';
  }
}
