import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const IspClientApp());
}

const String _apiBaseUrl = 'http://10.0.2.2:8000';

Future<DashboardData> fetchDashboardData() async {
  try {
    final uri = Uri.parse('$_apiBaseUrl/api/dashboard');
    final response = await http.get(uri).timeout(const Duration(seconds: 8));
    if (response.statusCode == 200) {
      final payload = jsonDecode(response.body) as Map<String, dynamic>;
      return DashboardData.fromJson(payload);
    }
    return DashboardData.fallback(
      reason: 'Server error ${response.statusCode}',
    );
  } catch (error) {
    return DashboardData.fallback(reason: error.toString());
  }
}

class IspClientApp extends StatelessWidget {
  const IspClientApp({super.key});

  @override
  Widget build(BuildContext context) {
    const primary = Color(0xFF0D6EFD);
    return MaterialApp(
      title: 'Client Dashboard',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: primary),
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF8F9FB),
        cardTheme: const CardThemeData(
          elevation: 0,
          margin: EdgeInsets.zero,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(12)),
          ),
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: primary,
          foregroundColor: Colors.white,
          centerTitle: false,
        ),
      ),
      home: const DashboardShell(),
    );
  }
}

class DashboardShell extends StatefulWidget {
  const DashboardShell({super.key});

  @override
  State<DashboardShell> createState() => _DashboardShellState();
}

class _DashboardShellState extends State<DashboardShell> {
  int _selectedIndex = 0;
  late Future<DashboardData> _dataFuture;

  final List<_NavItem> _navItems = const [
    _NavItem(label: 'Overview', icon: Icons.dashboard_rounded),
    _NavItem(label: 'Devices', icon: Icons.router_rounded),
    _NavItem(label: 'Service', icon: Icons.auto_graph_rounded),
    _NavItem(label: 'Alerts', icon: Icons.notifications_rounded),
  ];

  @override
  void initState() {
    super.initState();
    _dataFuture = fetchDashboardData();
  }

  void _refresh() {
    setState(() {
      _dataFuture = fetchDashboardData();
    });
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<DashboardData>(
      future: _dataFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }

        final data = snapshot.data ?? DashboardData.fallback(reason: 'No data');
        final pages = [
          OverviewPage(data: data),
          DevicesPage(data: data),
          ServicePage(data: data),
          AlertsPage(data: data),
        ];

        return Scaffold(
          appBar: AppBar(
            title: const Text('Client Dashboard'),
            actions: [
              IconButton(
                onPressed: _refresh,
                icon: const Icon(Icons.refresh_rounded),
                tooltip: 'Refresh',
              ),
              IconButton(
                onPressed: () {},
                icon: const Icon(Icons.search_rounded),
                tooltip: 'Search',
              ),
              Padding(
                padding: const EdgeInsets.only(right: 12),
                child: CircleAvatar(
                  backgroundColor: Colors.white.withValues(alpha: 0.2),
                  child: const Icon(Icons.person_rounded, color: Colors.white),
                ),
              ),
            ],
          ),
          body: SafeArea(child: pages[_selectedIndex]),
          bottomNavigationBar: NavigationBar(
            selectedIndex: _selectedIndex,
            destinations: [
              for (final item in _navItems)
                NavigationDestination(icon: Icon(item.icon), label: item.label),
            ],
            onDestinationSelected: (index) {
              setState(() {
                _selectedIndex = index;
              });
            },
          ),
        );
      },
    );
  }
}

class _NavItem {
  final String label;
  final IconData icon;

  const _NavItem({required this.label, required this.icon});
}

class OverviewPage extends StatelessWidget {
  final DashboardData data;

  const OverviewPage({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final summary = data.summary;
    final topDevices = data.topDevices;

    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        Text(
          'Overview',
          style: Theme.of(
            context,
          ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        if (data.isFallback) ...[
          OfflineBanner(message: data.fallbackReason),
          const SizedBox(height: 12),
        ] else
          const SizedBox(height: 4),
        GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 2,
            mainAxisSpacing: 12,
            crossAxisSpacing: 12,
            childAspectRatio: 1.9,
          ),
          itemCount: summary.length,
          itemBuilder: (context, index) => SummaryCard(metric: summary[index]),
        ),
        const SizedBox(height: 16),
        LayoutBuilder(
          builder: (context, constraints) {
            final wide = constraints.maxWidth >= 720;
            final chartCard = InfoPanel(
              title: 'Usage & Speed',
              child: SizedBox(
                height: 180,
                child: LineChartPlaceholder(
                  usage: data.chart.usage,
                  speed: data.chart.speed,
                ),
              ),
            );
            final topCard = InfoPanel(
              title: 'Top Devices (24h)',
              child: ListView.separated(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemBuilder: (context, index) =>
                    TopDeviceRow(device: topDevices[index]),
                separatorBuilder: (context, index) => const Divider(height: 16),
                itemCount: topDevices.length,
              ),
            );

            if (wide) {
              return Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(flex: 3, child: chartCard),
                  const SizedBox(width: 16),
                  Expanded(flex: 2, child: topCard),
                ],
              );
            }

            return Column(
              children: [chartCard, const SizedBox(height: 16), topCard],
            );
          },
        ),
      ],
    );
  }
}

class DevicesPage extends StatelessWidget {
  final DashboardData data;

  const DevicesPage({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final devices = data.devices;

    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        Text(
          'Devices',
          style: Theme.of(
            context,
          ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        if (data.isFallback) ...[
          OfflineBanner(message: data.fallbackReason),
          const SizedBox(height: 12),
        ] else
          const SizedBox(height: 4),
        InfoPanel(
          title: 'Devices',
          child: ListView.separated(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemBuilder: (context, index) => DeviceCard(device: devices[index]),
            separatorBuilder: (context, index) => const SizedBox(height: 12),
            itemCount: devices.length,
          ),
        ),
      ],
    );
  }
}

class ServicePage extends StatelessWidget {
  final DashboardData data;

  const ServicePage({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        Text(
          'Service',
          style: Theme.of(
            context,
          ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        if (data.isFallback) ...[
          OfflineBanner(message: data.fallbackReason),
          const SizedBox(height: 12),
        ] else
          const SizedBox(height: 4),
        LayoutBuilder(
          builder: (context, constraints) {
            final wide = constraints.maxWidth >= 720;
            final planCard = InfoPanel(
              title: 'Plan Details',
              child: Column(
                children: [
                  KeyValueRow(label: 'Plan', value: data.service.plan),
                  KeyValueRow(label: 'Next Bill', value: data.service.nextBill),
                  KeyValueRow(label: 'Due Date', value: data.service.dueDate),
                  KeyValueRow(
                    label: 'Billing Cycle',
                    value: data.service.billingCycle,
                  ),
                  KeyValueRow(
                    label: 'Router Model',
                    value: data.service.routerModel,
                  ),
                ],
              ),
            );

            final healthCard = InfoPanel(
              title: 'Service Health',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  ProgressRow(label: 'Uptime', value: data.service.uptime),
                  const SizedBox(height: 12),
                  ProgressRow(
                    label: 'Speed Consistency',
                    value: data.service.speedConsistency,
                  ),
                ],
              ),
            );

            if (wide) {
              return Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(child: planCard),
                  const SizedBox(width: 16),
                  Expanded(child: healthCard),
                ],
              );
            }

            return Column(
              children: [planCard, const SizedBox(height: 16), healthCard],
            );
          },
        ),
      ],
    );
  }
}

class AlertsPage extends StatelessWidget {
  final DashboardData data;

  const AlertsPage({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final alerts = data.alerts;

    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        Text(
          'Alerts',
          style: Theme.of(
            context,
          ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        if (data.isFallback) ...[
          OfflineBanner(message: data.fallbackReason),
          const SizedBox(height: 12),
        ] else
          const SizedBox(height: 4),
        InfoPanel(
          title: 'Alerts',
          child: ListView.separated(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemBuilder: (context, index) => AlertCard(alert: alerts[index]),
            separatorBuilder: (context, index) => const SizedBox(height: 12),
            itemCount: alerts.length,
          ),
        ),
      ],
    );
  }
}

class DashboardData {
  final List<SummaryMetric> summary;
  final List<TopDeviceItem> topDevices;
  final ChartData chart;
  final List<DeviceInfo> devices;
  final ServiceInfo service;
  final List<AlertInfo> alerts;
  final bool isFallback;
  final String fallbackReason;

  const DashboardData({
    required this.summary,
    required this.topDevices,
    required this.chart,
    required this.devices,
    required this.service,
    required this.alerts,
    required this.isFallback,
    required this.fallbackReason,
  });

  factory DashboardData.fromJson(Map<String, dynamic> json) {
    final summaryJson = (json['summary'] ?? {}) as Map<String, dynamic>;
    final total = summaryJson['total'] ?? 0;
    final online = summaryJson['online'] ?? 0;
    final degraded = summaryJson['degraded'] ?? 0;
    final alerts24 = summaryJson['alerts_24h'] ?? 0;

    final summary = <SummaryMetric>[
      SummaryMetric(
        'Total Devices',
        '$total',
        valueColor: const Color(0xFF212529),
      ),
      SummaryMetric('Online', '$online', valueColor: const Color(0xFF198754)),
      SummaryMetric(
        'Degraded',
        '$degraded',
        valueColor: const Color(0xFFFFC107),
      ),
      SummaryMetric(
        'Alerts (24h)',
        '$alerts24',
        valueColor: const Color(0xFFDC3545),
      ),
    ];

    final topDevicesJson = (json['top_devices'] ?? []) as List<dynamic>;
    final topDevices = topDevicesJson
        .map(
          (item) => TopDeviceItem(
            item['name']?.toString() ?? 'Unknown',
            item['usage']?.toString() ?? '0 GB',
          ),
        )
        .toList();

    final chartJson = (json['chart'] ?? {}) as Map<String, dynamic>;
    final usage = (chartJson['usage'] ?? []) as List<dynamic>;
    final speed = (chartJson['speed'] ?? []) as List<dynamic>;
    final chart = ChartData(
      usage: usage.map((e) => (e as num).toDouble()).toList(),
      speed: speed.map((e) => (e as num).toDouble()).toList(),
    );

    final devicesJson = (json['devices'] ?? []) as List<dynamic>;
    final devices = devicesJson
        .map(
          (item) => DeviceInfo(
            name: item['name']?.toString() ?? 'Device',
            room: item['room']?.toString() ?? 'Unknown',
            status: _deviceStateFromString(item['status']?.toString()),
            lastSeen: item['last_seen']?.toString() ?? '-',
            firmware: item['firmware']?.toString() ?? '-',
            usage24h: item['usage_24h']?.toString() ?? '-',
            model: item['model']?.toString() ?? '-',
          ),
        )
        .toList();

    final serviceJson = (json['service'] ?? {}) as Map<String, dynamic>;
    final service = ServiceInfo(
      plan: serviceJson['plan']?.toString() ?? 'Home Fiber 50',
      nextBill: serviceJson['next_bill']?.toString() ?? '\$45.00',
      dueDate: serviceJson['due_date']?.toString() ?? 'Feb 5, 2026',
      billingCycle: serviceJson['billing_cycle']?.toString() ?? 'Monthly',
      routerModel: serviceJson['router_model']?.toString() ?? 'XR-500',
      uptime: (serviceJson['uptime'] ?? 0.98).toDouble(),
      speedConsistency: (serviceJson['speed_consistency'] ?? 0.86).toDouble(),
    );

    final alertsJson = (json['alerts'] ?? []) as List<dynamic>;
    final alerts = alertsJson
        .map(
          (item) => AlertInfo(
            title: item['title']?.toString() ?? 'Alert',
            message: item['message']?.toString() ?? 'No details',
            time: item['time']?.toString() ?? '-',
            severity: _severityFromString(item['severity']?.toString()),
          ),
        )
        .toList();

    return DashboardData(
      summary: summary,
      topDevices: topDevices,
      chart: chart,
      devices: devices,
      service: service,
      alerts: alerts,
      isFallback: false,
      fallbackReason: '',
    );
  }

  static DashboardData fallback({required String reason}) {
    return DashboardData(
      summary: const [
        SummaryMetric('Total Devices', '0', valueColor: Color(0xFF212529)),
        SummaryMetric('Online', '0', valueColor: Color(0xFF198754)),
        SummaryMetric('Degraded', '0', valueColor: Color(0xFFFFC107)),
        SummaryMetric('Alerts (24h)', '0', valueColor: Color(0xFFDC3545)),
      ],
      topDevices: const [],
      chart: const ChartData(usage: [0, 0, 0, 0], speed: [0, 0, 0, 0]),
      devices: const [],
      service: const ServiceInfo(
        plan: 'Home Fiber 50',
        nextBill: '\$45.00',
        dueDate: 'Feb 5, 2026',
        billingCycle: 'Monthly',
        routerModel: 'XR-500',
        uptime: 0.0,
        speedConsistency: 0.0,
      ),
      alerts: const [],
      isFallback: true,
      fallbackReason: reason,
    );
  }
}

class ChartData {
  final List<double> usage;
  final List<double> speed;

  const ChartData({required this.usage, required this.speed});
}

class ServiceInfo {
  final String plan;
  final String nextBill;
  final String dueDate;
  final String billingCycle;
  final String routerModel;
  final double uptime;
  final double speedConsistency;

  const ServiceInfo({
    required this.plan,
    required this.nextBill,
    required this.dueDate,
    required this.billingCycle,
    required this.routerModel,
    required this.uptime,
    required this.speedConsistency,
  });
}

class SummaryMetric {
  final String label;
  final String value;
  final Color valueColor;

  const SummaryMetric(this.label, this.value, {required this.valueColor});
}

class TopDeviceItem {
  final String name;
  final String usage;

  const TopDeviceItem(this.name, this.usage);
}

enum DeviceState { online, degraded, offline }

class DeviceInfo {
  final String name;
  final String room;
  final DeviceState status;
  final String lastSeen;
  final String firmware;
  final String usage24h;
  final String model;

  const DeviceInfo({
    required this.name,
    required this.room,
    required this.status,
    required this.lastSeen,
    required this.firmware,
    required this.usage24h,
    required this.model,
  });
}

enum AlertSeverity { high, medium, low }

class AlertInfo {
  final String title;
  final String message;
  final String time;
  final AlertSeverity severity;

  const AlertInfo({
    required this.title,
    required this.message,
    required this.time,
    required this.severity,
  });
}

class SummaryCard extends StatelessWidget {
  final SummaryMetric metric;

  const SummaryCard({super.key, required this.metric});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              metric.label,
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(color: Colors.black54),
            ),
            const SizedBox(height: 8),
            Text(
              metric.value,
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.bold,
                color: metric.valueColor,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class InfoPanel extends StatelessWidget {
  final String title;
  final Widget child;

  const InfoPanel({super.key, required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 12),
            child,
          ],
        ),
      ),
    );
  }
}

class OfflineBanner extends StatelessWidget {
  final String message;

  const OfflineBanner({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF3CD),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFFFE69C)),
      ),
      child: Row(
        children: [
          const Icon(Icons.info_outline, color: Color(0xFFB54708)),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'Offline data: $message',
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(color: const Color(0xFF7A2E0E)),
            ),
          ),
        ],
      ),
    );
  }
}

class TopDeviceRow extends StatelessWidget {
  final TopDeviceItem device;

  const TopDeviceRow({super.key, required this.device});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(device.name, style: Theme.of(context).textTheme.bodyMedium),
        BadgePill(label: device.usage, color: const Color(0xFF0D6EFD)),
      ],
    );
  }
}

class DeviceCard extends StatelessWidget {
  final DeviceInfo device;

  const DeviceCard({super.key, required this.device});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  device.name,
                  style: Theme.of(
                    context,
                  ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold),
                ),
                StatusBadge(status: device.status),
              ],
            ),
            const SizedBox(height: 8),
            LabelValueRow(label: 'Room', value: device.room),
            LabelValueRow(label: 'Last Seen', value: device.lastSeen),
            LabelValueRow(label: 'Firmware', value: device.firmware),
            LabelValueRow(label: 'Usage (24h)', value: device.usage24h),
            LabelValueRow(label: 'Model', value: device.model),
          ],
        ),
      ),
    );
  }
}

class AlertCard extends StatelessWidget {
  final AlertInfo alert;

  const AlertCard({super.key, required this.alert});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  alert.title,
                  style: Theme.of(
                    context,
                  ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold),
                ),
                SeverityBadge(severity: alert.severity),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              alert.time,
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(color: Colors.black54),
            ),
            const SizedBox(height: 6),
            Text(alert.message, style: Theme.of(context).textTheme.bodyMedium),
          ],
        ),
      ),
    );
  }
}

class LabelValueRow extends StatelessWidget {
  final String label;
  final String value;

  const LabelValueRow({super.key, required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: Theme.of(
              context,
            ).textTheme.bodySmall?.copyWith(color: Colors.black54),
          ),
          Flexible(
            child: Text(
              value,
              textAlign: TextAlign.right,
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
        ],
      ),
    );
  }
}

class KeyValueRow extends StatelessWidget {
  final String label;
  final String value;

  const KeyValueRow({super.key, required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: Theme.of(
              context,
            ).textTheme.bodySmall?.copyWith(color: Colors.black54),
          ),
          Text(
            value,
            style: Theme.of(
              context,
            ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }
}

class ProgressRow extends StatelessWidget {
  final String label;
  final double value;

  const ProgressRow({super.key, required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final display = (value * 100).toStringAsFixed(0);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: Theme.of(context).textTheme.bodySmall),
            Text('$display%'),
          ],
        ),
        const SizedBox(height: 6),
        LinearProgressIndicator(
          value: value,
          minHeight: 8,
          color: Theme.of(context).colorScheme.primary,
          backgroundColor: Colors.black12,
          borderRadius: BorderRadius.circular(12),
        ),
      ],
    );
  }
}

class BadgePill extends StatelessWidget {
  final String label;
  final Color color;

  const BadgePill({super.key, required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Text(
        label,
        style: Theme.of(context).textTheme.labelSmall?.copyWith(color: color),
      ),
    );
  }
}

class StatusBadge extends StatelessWidget {
  final DeviceState status;

  const StatusBadge({super.key, required this.status});

  @override
  Widget build(BuildContext context) {
    final color = _statusColor(status);
    final label = _statusLabel(status);
    return BadgePill(label: label, color: color);
  }
}

class SeverityBadge extends StatelessWidget {
  final AlertSeverity severity;

  const SeverityBadge({super.key, required this.severity});

  @override
  Widget build(BuildContext context) {
    final color = _severityColor(severity);
    final label = _severityLabel(severity);
    return BadgePill(label: label, color: color);
  }
}

class LineChartPlaceholder extends StatelessWidget {
  final List<double> usage;
  final List<double> speed;

  const LineChartPlaceholder({
    super.key,
    required this.usage,
    required this.speed,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _LineChartPainter(usage: usage, speed: speed),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 12),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.end,
          children: const [
            _LegendDot(label: 'Usage', color: Color(0xFFDC3545)),
            SizedBox(width: 12),
            _LegendDot(label: 'Speed', color: Color(0xFF0D6EFD)),
          ],
        ),
      ),
    );
  }
}

class _LegendDot extends StatelessWidget {
  final String label;
  final Color color;

  const _LegendDot({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 6),
        Text(label, style: Theme.of(context).textTheme.labelSmall),
      ],
    );
  }
}

class _LineChartPainter extends CustomPainter {
  final List<double> usage;
  final List<double> speed;

  _LineChartPainter({required this.usage, required this.speed});

  @override
  void paint(Canvas canvas, Size size) {
    final gridPaint = Paint()
      ..color = Colors.black12
      ..strokeWidth = 1;

    final chartHeight = size.height - 28;
    final chartWidth = size.width - 16;
    final origin = const Offset(8, 8);

    for (int i = 0; i < 4; i++) {
      final y = origin.dy + (chartHeight / 3) * i;
      canvas.drawLine(
        Offset(origin.dx, y),
        Offset(origin.dx + chartWidth, y),
        gridPaint,
      );
    }

    _drawLine(
      canvas,
      origin,
      chartWidth,
      chartHeight,
      usage,
      const Color(0xFFDC3545),
    );
    _drawLine(
      canvas,
      origin,
      chartWidth,
      chartHeight,
      speed,
      const Color(0xFF0D6EFD),
    );
  }

  void _drawLine(
    Canvas canvas,
    Offset origin,
    double width,
    double height,
    List<double> values,
    Color color,
  ) {
    if (values.length < 2) {
      return;
    }
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.2;

    final path = Path();
    final maxValue = values.reduce((a, b) => a > b ? a : b);
    final minValue = values.reduce((a, b) => a < b ? a : b);
    final range = (maxValue - minValue).abs() < 0.001 ? 1 : maxValue - minValue;

    for (int i = 0; i < values.length; i++) {
      final dx = origin.dx + (width / (values.length - 1)) * i;
      final normalized = (values[i] - minValue) / range;
      final dy = origin.dy + height - (normalized * height);
      if (i == 0) {
        path.moveTo(dx, dy);
      } else {
        path.lineTo(dx, dy);
      }
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

DeviceState _deviceStateFromString(String? value) {
  switch (value?.toLowerCase()) {
    case 'online':
      return DeviceState.online;
    case 'degraded':
      return DeviceState.degraded;
    case 'offline':
      return DeviceState.offline;
    default:
      return DeviceState.online;
  }
}

AlertSeverity _severityFromString(String? value) {
  switch (value?.toLowerCase()) {
    case 'high':
      return AlertSeverity.high;
    case 'medium':
      return AlertSeverity.medium;
    case 'low':
      return AlertSeverity.low;
    default:
      return AlertSeverity.low;
  }
}

Color _statusColor(DeviceState status) {
  switch (status) {
    case DeviceState.online:
      return const Color(0xFF198754);
    case DeviceState.degraded:
      return const Color(0xFFFFC107);
    case DeviceState.offline:
      return const Color(0xFF6C757D);
  }
}

String _statusLabel(DeviceState status) {
  switch (status) {
    case DeviceState.online:
      return 'online';
    case DeviceState.degraded:
      return 'degraded';
    case DeviceState.offline:
      return 'offline';
  }
}

Color _severityColor(AlertSeverity severity) {
  switch (severity) {
    case AlertSeverity.high:
      return const Color(0xFFDC3545);
    case AlertSeverity.medium:
      return const Color(0xFFFFC107);
    case AlertSeverity.low:
      return const Color(0xFF6C757D);
  }
}

String _severityLabel(AlertSeverity severity) {
  switch (severity) {
    case AlertSeverity.high:
      return 'high';
    case AlertSeverity.medium:
      return 'medium';
    case AlertSeverity.low:
      return 'low';
  }
}
