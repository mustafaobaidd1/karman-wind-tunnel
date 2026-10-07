/** Inline SVG icons (24×24 grid unless noted). Drawn for this project. */

const s = (body: string, viewBox = '0 0 24 24') =>
  `<svg viewBox="${viewBox}" aria-hidden="true" focusable="false" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;

export const icons = {
  brush: s(
    '<path d="M14.5 4.5l5 5-7.8 7.8-5-5z"/><path d="M6.7 12.3l5 5c-1.2 2.4-3.9 3.9-8 3.7.4-3.8 1.4-6.6 3-8.7z" fill="currentColor" stroke="none"/>',
  ),
  eraser: s('<path d="M4 15.5l9-9a2 2 0 0 1 2.8 0l3.2 3.2a2 2 0 0 1 0 2.8L12 19.5H7.5z"/><path d="M9 10.5l6 6"/><path d="M12 19.5h8"/>'),
  move: s(
    '<path d="M12 3v18M3 12h18"/><path d="M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3"/>',
  ),
  probe: s('<circle cx="12" cy="12" r="6.5"/><path d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4"/><circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none"/>'),
  play: s('<path d="M8 5.5v13l10.5-6.5z" fill="currentColor"/>'),
  pause: s('<path d="M8 5.5v13M16 5.5v13" stroke-width="2.6"/>'),
  step: s('<path d="M6.5 6v12l8.5-6z" fill="currentColor"/><path d="M18 6v12" stroke-width="2.2"/>'),
  reset: s('<path d="M4.5 12a7.5 7.5 0 1 0 2.4-5.5"/><path d="M4 4.5v4h4"/>'),
  trash: s('<path d="M4.5 7h15M10 4h4M6.5 7l1 13h9l1-13"/><path d="M10 11v5.5M14 11v5.5"/>'),
  info: s('<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>'),
  gauge: s('<path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 18l4.5-6"/><path d="M8 18h8"/>'),
  menu: s('<path d="M4 7h16M4 12h16M4 17h16"/>'),
  close: s('<path d="M6 6l12 12M18 6L6 18"/>'),
  eye: s('<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>'),
  warn: s('<path d="M12 3.5l9.5 16.5h-19z"/><path d="M12 10v4.5M12 17.2v.3"/>'),
  plus: s('<path d="M12 5v14M5 12h14"/>'),
  stop: s('<rect x="6.5" y="6.5" width="11" height="11" rx="1.5" fill="currentColor"/>'),
  exit: s('<path d="M10 5H5.5v14H10"/><path d="M14 8l4 4-4 4M18 12H9"/>'),
  // Preset glyphs on a 30×20 canvas, filled.
  circle: `<svg viewBox="0 0 30 20" aria-hidden="true" focusable="false"><circle cx="15" cy="10" r="6.5" fill="currentColor"/></svg>`,
  square: `<svg viewBox="0 0 30 20" aria-hidden="true" focusable="false"><rect x="9" y="4" width="12" height="12" fill="currentColor"/></svg>`,
  plate: `<svg viewBox="0 0 30 20" aria-hidden="true" focusable="false"><rect x="13.6" y="2" width="2.8" height="16" fill="currentColor"/></svg>`,
  airfoil: `<svg viewBox="0 0 30 20" aria-hidden="true" focusable="false"><path d="M3 11.3C3.6 6.8 9 5.7 15 6.6c4.8.7 8.9 2.9 12 5.4-4.4-.4-8.4-.2-12.1.4C9.7 13.2 4.2 14 3 11.3z" fill="currentColor"/></svg>`,
  tandem: `<svg viewBox="0 0 30 20" aria-hidden="true" focusable="false"><circle cx="8" cy="10" r="4.4" fill="currentColor"/><circle cx="22" cy="10" r="4.4" fill="currentColor"/></svg>`,
  car: `<svg viewBox="0 0 30 20" aria-hidden="true" focusable="false"><path d="M2 13.6v-2.2l2.3-.6 5-1.2 3.6-3.3h7.8l3.8 3.4 3.6.8.9 1.6v1.5z" fill="currentColor"/><circle cx="8.3" cy="14.3" r="2.3" fill="currentColor"/><circle cx="21.7" cy="14.3" r="2.3" fill="currentColor"/></svg>`,
  splitter: `<svg viewBox="0 0 30 20" aria-hidden="true" focusable="false"><circle cx="10" cy="10" r="5.5" fill="currentColor"/><rect x="14" y="9" width="12" height="2" fill="currentColor"/></svg>`,
  building: `<svg viewBox="0 0 30 20" aria-hidden="true" focusable="false"><rect x="11" y="3" width="8" height="15" fill="currentColor"/><path d="M4 18.5h22" stroke="currentColor" stroke-width="1.2"/></svg>`,
} as const;
