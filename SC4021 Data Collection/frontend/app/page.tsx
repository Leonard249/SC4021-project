'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { verifySecret } from '@/app/lib/api';

export default function AuthPage() {
  const [secretKey, setSecretKey] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!secretKey.trim()) {
      setError('Please enter a secret key.');
      return;
    }

    setLoading(true);
    setError('');

    const isValid = await verifySecret(secretKey);

    if (isValid) {
      // Store a flag or simply rely on state (in a real app we'd use cookies/JWT, but here localStorage is simple)
      if (typeof window !== 'undefined') {
        localStorage.setItem('auth_secret', secretKey);
      }
      router.push('/labeler');
    } else {
      setError('Invalid secret key.');
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-gradient-to-br from-indigo-900 via-purple-900 to-slate-900 p-4">
      <div className="w-full max-w-md rounded-2xl bg-white/10 p-8 backdrop-blur-lg shadow-2xl border border-white/20">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-extrabold tracking-tight text-white mb-2">
            Data Labeling
          </h1>
          <p className="text-sm text-purple-200">
            Enter your secret key to continue
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="secretKey" className="sr-only">
              Secret Key
            </label>
            <input
              id="secretKey"
              type="password"
              value={secretKey}
              onChange={(e) => setSecretKey(e.target.value)}
              placeholder="••••••••"
              className="w-full rounded-xl border border-white/20 bg-black/20 px-4 py-3 text-white placeholder-gray-400 focus:border-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 transition-all"
              required
            />
          </div>

          {error && (
            <div className="rounded-lg bg-red-500/20 p-3 text-sm text-red-200 border border-red-500/50">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-xl bg-purple-600 px-4 py-3 font-semibold text-white shadow-lg hover:bg-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-gray-900 disabled:opacity-50 disabled:cursor-not-allowed transition-all active:scale-[0.98]"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="h-5 w-5 animate-spin text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Verifying...
              </span>
            ) : (
              'Enter Workspace'
            )}
          </button>
        </form>
      </div>
    </main>
  );
}
