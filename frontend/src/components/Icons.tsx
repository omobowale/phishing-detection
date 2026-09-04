import type { SVGProps } from 'react';

type IconProps = SVGProps<SVGSVGElement>;

function Icon({ children, ...props }: IconProps) {
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{children}</svg>;
}

export const ShieldIcon = (props: IconProps) => <Icon {...props}><path d="M12 3 4.8 6v5.2c0 4.5 2.9 8.3 7.2 9.8 4.3-1.5 7.2-5.3 7.2-9.8V6L12 3Z"/><path d="m8.8 12 2.1 2.1 4.5-4.5"/></Icon>;
export const ScanIcon = (props: IconProps) => <Icon {...props}><path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M8 21H5a2 2 0 0 1-2-2v-3M16 21h3a2 2 0 0 0 2-2v-3"/><path d="M7.5 12h9M12 7.5v9"/></Icon>;
export const HistoryIcon = (props: IconProps) => <Icon {...props}><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/></Icon>;
export const ListIcon = (props: IconProps) => <Icon {...props}><path d="M8 6h12M8 12h12M8 18h12"/><path d="m3.5 6 .8.8L6 5M3.5 12l.8.8L6 11M3.5 18l.8.8L6 17"/></Icon>;
export const ChartIcon = (props: IconProps) => <Icon {...props}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></Icon>;
export const LogOutIcon = (props: IconProps) => <Icon {...props}><path d="M10 17l5-5-5-5M15 12H3M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/></Icon>;
export const LinkIcon = (props: IconProps) => <Icon {...props}><path d="M10 13a5 5 0 0 0 7.5.5l2-2a5 5 0 0 0-7-7l-1.1 1.1"/><path d="M14 11a5 5 0 0 0-7.5-.5l-2 2a5 5 0 0 0 7 7l1.1-1.1"/></Icon>;
export const MailIcon = (props: IconProps) => <Icon {...props}><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/></Icon>;
export const ArrowIcon = (props: IconProps) => <Icon {...props}><path d="M5 12h14M14 7l5 5-5 5"/></Icon>;
export const CheckIcon = (props: IconProps) => <Icon {...props}><path d="m5 12 4 4L19 6"/></Icon>;
export const AlertIcon = (props: IconProps) => <Icon {...props}><path d="M10.3 3.9 2.4 18a2 2 0 0 0 1.8 3h15.6a2 2 0 0 0 1.8-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4M12 17h.01"/></Icon>;
export const ClockIcon = (props: IconProps) => <Icon {...props}><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></Icon>;
export const LockIcon = (props: IconProps) => <Icon {...props}><rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></Icon>;
export const TrashIcon = (props: IconProps) => <Icon {...props}><path d="M4 7h16M9 7V4h6v3M7 7l1 14h8l1-14M10 11v6M14 11v6"/></Icon>;
export const SunIcon = (props: IconProps) => <Icon {...props}><circle cx="12" cy="12" r="4"/><path d="M12 3v2M12 19v2M5 5l1.4 1.4M17.6 17.6 19 19M3 12h2M19 12h2M5 19l1.4-1.4M17.6 6.4 19 5"/></Icon>;
export const MoonIcon = (props: IconProps) => <Icon {...props}><path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z"/></Icon>;
export const CloseIcon = (props: IconProps) => <Icon {...props}><path d="M6 6l12 12M18 6 6 18"/></Icon>;
