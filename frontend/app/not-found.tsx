export default function NotFound() {
    console.log('⚠️ Custom 404 rendered – we reached Next.js');
    return (
        <main className="flex min-h-screen items-center justify-center bg-red-900 text-white">
            <h2 className="text-2xl font-bold">Page not found (404)</h2>
        </main>
    );
}
