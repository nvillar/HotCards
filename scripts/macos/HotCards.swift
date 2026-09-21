import AppKit
import Foundation

struct LauncherConfiguration: Decodable {
    let RuntimeDirectory: String
}

enum LauncherError: LocalizedError {
    case invalidInstallation(String)
    case applicationFailed(Int32)

    var errorDescription: String? {
        switch self {
        case .invalidInstallation(let message):
            return message
        case .applicationFailed(let status):
            return "HotCards exited unexpectedly (status \(status))."
        }
    }
}

let logDirectory = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library/Logs/HotCards", isDirectory: true)
let logURL = logDirectory.appendingPathComponent("launcher.log")

do {
    guard let configurationURL = Bundle.main.url(forResource: "Launcher", withExtension: "plist") else {
        throw LauncherError.invalidInstallation("The launcher configuration is missing. Reinstall HotCards.")
    }
    let configuration = try PropertyListDecoder().decode(
        LauncherConfiguration.self, from: Data(contentsOf: configurationURL)
    )
    let runtime = URL(fileURLWithPath: configuration.RuntimeDirectory, isDirectory: true)
    let executable = runtime.appendingPathComponent(".venv/bin/hotcards")
    guard FileManager.default.isExecutableFile(atPath: executable.path) else {
        throw LauncherError.invalidInstallation(
            "The HotCards Python environment is missing or unreadable. Reinstall the launcher."
        )
    }
    if CommandLine.arguments.contains("--check") {
        print("HotCards launcher configuration is valid.")
        exit(0)
    }
    try FileManager.default.createDirectory(at: logDirectory, withIntermediateDirectories: true)
    if !FileManager.default.fileExists(atPath: logURL.path) {
        guard FileManager.default.createFile(atPath: logURL.path, contents: nil) else {
            throw LauncherError.invalidInstallation("Could not create \(logURL.path).")
        }
    }
    let log = try FileHandle(forWritingTo: logURL)
    defer { try? log.close() }
    try log.seekToEnd()
    try log.write(contentsOf: Data("\n--- HotCards launch \(Date()) ---\n".utf8))
    let process = Process()
    process.executableURL = executable
    process.currentDirectoryURL = runtime
    var environment = ProcessInfo.processInfo.environment
    environment["PYTHONUNBUFFERED"] = "1"
    // Never inherit an unrelated Python environment from the launching process.
    for key in ["PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"] {
        environment.removeValue(forKey: key)
    }
    process.environment = environment
    process.standardOutput = log
    process.standardError = log
    try process.run()
    process.waitUntilExit()
    if process.terminationStatus != 0 {
        throw LauncherError.applicationFailed(process.terminationStatus)
    }
} catch {
    fputs("HotCards launcher: \(error.localizedDescription)\n", stderr)
    if !CommandLine.arguments.contains("--check") {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let alert = NSAlert()
        alert.messageText = "HotCards Could Not Run"
        alert.informativeText = "\(error.localizedDescription)\n\nDiagnostic log: \(logURL.path)"
        alert.alertStyle = .critical
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Open Logs")
        app.activate(ignoringOtherApps: true)
        if alert.runModal() == .alertSecondButtonReturn {
            NSWorkspace.shared.open(logDirectory)
        }
    }
    exit(1)
}
