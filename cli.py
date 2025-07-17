#!/usr/bin/env python3
"""
Command-line interface for the Research RAG System
"""

import argparse
import sys
import os
from typing import List, Dict
import json

from rag_system import ResearchRAGSystem
from config import Config

def main():
    parser = argparse.ArgumentParser(
        description="Research RAG System - Process PDFs and answer questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py process-azure                    # Process PDFs from Azure
  python cli.py process-local /path/to/pdfs      # Process local PDFs
  python cli.py process-local-storage            # Process PDFs from local storage
  python cli.py add-pdf /path/to/paper.pdf      # Add PDF to local storage
  python cli.py ask "What are the main findings?" # Ask a question
  python cli.py summary "machine learning"       # Generate summary
  python cli.py stats                           # Show system stats
  python cli.py list-local-pdfs                 # List PDFs in local storage
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process Azure PDFs command
    process_azure_parser = subparsers.add_parser(
        'process-azure',
        help='Process PDFs from Azure Blob Storage'
    )
    process_azure_parser.add_argument(
        '--download-local',
        action='store_true',
        help='Download PDFs to local storage before processing'
    )
    
    # Process local PDFs command
    process_local_parser = subparsers.add_parser(
        'process-local',
        help='Process PDFs from local directory'
    )
    process_local_parser.add_argument(
        'directory',
        help='Path to directory containing PDF files'
    )
    
    # Process local storage PDFs command
    process_local_storage_parser = subparsers.add_parser(
        'process-local-storage',
        help='Process PDFs from local storage'
    )
    
    # Add PDF to local storage command
    add_pdf_parser = subparsers.add_parser(
        'add-pdf',
        help='Add a PDF file to local storage'
    )
    add_pdf_parser.add_argument(
        'file_path',
        help='Path to the PDF file to add'
    )
    add_pdf_parser.add_argument(
        '--no-organize',
        action='store_true',
        help='Do not organize files by date/type'
    )
    
    # List local PDFs command
    list_local_pdfs_parser = subparsers.add_parser(
        'list-local-pdfs',
        help='List PDFs in local storage'
    )
    
    # Delete local PDF command
    delete_local_pdf_parser = subparsers.add_parser(
        'delete-local-pdf',
        help='Delete a PDF from local storage'
    )
    delete_local_pdf_parser.add_argument(
        'file_name',
        help='Name of the PDF file to delete'
    )
    
    # Cleanup backups command
    cleanup_backups_parser = subparsers.add_parser(
        'cleanup-backups',
        help='Clean up old backup files in local storage'
    )
    cleanup_backups_parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to keep backups (default: 30)'
    )
    
    # Ask question command
    ask_parser = subparsers.add_parser(
        'ask',
        help='Ask a question about the research papers'
    )
    ask_parser.add_argument(
        'question',
        help='The question to ask'
    )
    ask_parser.add_argument(
        '--no-sources',
        action='store_true',
        help='Exclude source information from response'
    )
    
    # Summary command
    summary_parser = subparsers.add_parser(
        'summary',
        help='Generate research summary'
    )
    summary_parser.add_argument(
        'topic',
        nargs='?',
        default=None,
        help='Specific topic to focus on (optional)'
    )
    
    # Stats command
    stats_parser = subparsers.add_parser(
        'stats',
        help='Show system statistics'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize RAG system
        rag_system = ResearchRAGSystem()
        
        if args.command == 'process-azure':
            process_azure_pdfs(rag_system, args.download_local)
        elif args.command == 'process-local':
            process_local_pdfs(rag_system, args.directory)
        elif args.command == 'process-local-storage':
            process_local_storage_pdfs(rag_system)
        elif args.command == 'add-pdf':
            add_pdf_to_local_storage(rag_system, args.file_path, not args.no_organize)
        elif args.command == 'list-local-pdfs':
            list_local_pdfs(rag_system)
        elif args.command == 'delete-local-pdf':
            delete_local_pdf(rag_system, args.file_name)
        elif args.command == 'cleanup-backups':
            cleanup_backups(rag_system, args.days)
        elif args.command == 'ask':
            ask_question(rag_system, args.question, not args.no_sources)
        elif args.command == 'summary':
            generate_summary(rag_system, args.topic)
        elif args.command == 'stats':
            show_stats(rag_system)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def process_azure_pdfs(rag_system, download_local):
    """Process PDFs from Azure Blob Storage"""
    print("🔄 Processing PDFs from Azure Blob Storage...")
    
    result = rag_system.process_azure_pdfs(download_local)
    
    if result['success']:
        print(f"✅ Successfully processed {result['pdfs_processed']} PDFs!")
        print(f"📊 Added {result['chunks_added']} text chunks to vector store")
    else:
        print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

def process_local_pdfs(rag_system, directory):
    """Process PDFs from local directory"""
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        sys.exit(1)
    
    print(f"🔄 Processing PDFs from local directory: {directory}")
    
    result = rag_system.process_local_pdfs(directory)
    
    if result['success']:
        print(f"✅ Successfully processed {result['pdfs_processed']} PDFs!")
        print(f"📊 Added {result['chunks_added']} text chunks to vector store")
    else:
        print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

def process_local_storage_pdfs(rag_system):
    """Process PDFs from local storage"""
    print("🔄 Processing PDFs from local storage...")
    
    result = rag_system.process_local_storage_pdfs()
    
    if result['success']:
        print(f"✅ Successfully processed {result['pdfs_processed']} PDFs!")
        print(f"📊 Added {result['chunks_added']} text chunks to vector store")
    else:
        print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

def add_pdf_to_local_storage(rag_system, file_path, organize):
    """Add a PDF file to local storage"""
    print(f"📄 Adding PDF to local storage: {file_path}")
    
    result = rag_system.add_pdf_to_local_storage(file_path, organize)
    
    if result['success']:
        print(f"✅ Successfully added {result['file_name']} to local storage")
        print(f"📁 Location: {result['local_path']}")
        print(f"📏 Size: {result['file_size'] / (1024*1024):.2f} MB")
    else:
        print(f"❌ Failed to add PDF: {result.get('error', 'Unknown error')}")
        sys.exit(1)

def list_local_pdfs(rag_system):
    """List PDFs in local storage"""
    print("📚 Listing PDFs in local storage...")
    
    pdfs = rag_system.get_local_storage_pdfs()
    
    if not pdfs:
        print("📭 No PDFs found in local storage")
        return
    
    print(f"\n📊 Found {len(pdfs)} PDFs:")
    print("-" * 80)
    
    for i, pdf in enumerate(pdfs, 1):
        if 'error' not in pdf:
            size_mb = pdf['size'] / (1024 * 1024)
            modified = pdf['modified'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"{i:2d}. {pdf['name']}")
            print(f"    📁 Path: {pdf['relative_path']}")
            print(f"    📏 Size: {size_mb:.2f} MB")
            print(f"    📅 Modified: {modified}")
            print()
        else:
            print(f"{i:2d}. {pdf['name']} (Error: {pdf['error']})")
            print()

def delete_local_pdf(rag_system, file_name):
    """Delete a PDF from local storage"""
    print(f"🗑️ Deleting PDF from local storage: {file_name}")
    
    result = rag_system.delete_local_pdf(file_name)
    
    if result['success']:
        print(f"✅ Successfully deleted {file_name}")
        if result.get('backup_created'):
            print("📦 Backup created before deletion")
    else:
        print(f"❌ Failed to delete PDF: {result.get('error', 'Unknown error')}")
        sys.exit(1)

def cleanup_backups(rag_system, days):
    """Clean up old backup files"""
    print(f"🧹 Cleaning up backups older than {days} days...")
    
    result = rag_system.cleanup_local_backups(days)
    
    if result['success']:
        print(f"✅ Cleaned up {result['files_deleted']} old backup files")
    else:
        print(f"❌ Cleanup failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

def ask_question(rag_system, question, include_sources):
    """Ask a question about the research papers"""
    print(f"🔍 Searching for answer to: {question}")
    
    result = rag_system.ask_question(question, include_sources)
    
    if result['success']:
        print("\n" + "="*50)
        print("📝 ANSWER")
        print("="*50)
        print(result['answer'])
        print("="*50)
        
        if include_sources and result['sources']:
            print("\n📚 SOURCES")
            print("-"*30)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['file_name']} (Score: {source['similarity_score']:.3f})")
        
        print(f"\n📊 Metadata: {result['num_sources']} sources, {result['context_length']:,} chars")
    else:
        print(f"❌ Failed to generate answer: {result.get('answer', 'Unknown error')}")
        sys.exit(1)

def generate_summary(rag_system, topic):
    """Generate research summary"""
    if topic:
        print(f"📋 Generating summary for topic: {topic}")
    else:
        print("📋 Generating general research summary")
    
    result = rag_system.summarize_research_findings(topic)
    
    if result['success']:
        print("\n" + "="*50)
        print("📝 RESEARCH SUMMARY")
        print("="*50)
        print(result['summary'])
        print("="*50)
        
        if result['sources_used']:
            print(f"\n📚 Sources used: {len(result['sources_used'])}")
    else:
        print(f"❌ Failed to generate summary: {result.get('error', 'Unknown error')}")
        sys.exit(1)

def show_stats(rag_system):
    """Show system statistics"""
    print("📊 Loading system statistics...")
    
    stats = rag_system.get_system_stats()
    
    if 'error' in stats:
        print(f"❌ Failed to load statistics: {stats['error']}")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("📊 SYSTEM STATISTICS")
    print("="*50)
    
    # Vector Store Stats
    print("\n🗄️ VECTOR STORE")
    print("-"*20)
    vector_stats = stats['vector_store']
    print(f"Total Documents: {vector_stats.get('total_documents', 0)}")
    print(f"Unique Files: {vector_stats.get('unique_files', 0)}")
    print(f"Estimated Chunks: {vector_stats.get('estimated_total_chunks', 0)}")
    
    # Azure Storage Stats
    print("\n☁️ AZURE BLOB STORAGE")
    print("-"*25)
    azure_stats = stats['azure_storage']
    if azure_stats.get('enabled') is False:
        print("Status: Disabled")
    elif 'error' in azure_stats:
        print(f"Error: {azure_stats['error']}")
    else:
        print(f"Total PDFs: {azure_stats.get('total_pdfs', 0)}")
        pdf_names = azure_stats.get('pdf_names', [])
        if pdf_names:
            print("PDF Files:")
            for name in pdf_names[:10]:  # Show first 10
                print(f"  - {name}")
            if len(pdf_names) > 10:
                print(f"  ... and {len(pdf_names) - 10} more")
    
    # Local Storage Stats
    print("\n💾 LOCAL STORAGE")
    print("-"*15)
    local_stats = stats['local_storage']
    if local_stats.get('enabled') is False:
        print("Status: Disabled")
    elif 'error' in local_stats:
        print(f"Error: {local_stats['error']}")
    else:
        print(f"Total Files: {local_stats.get('total_files', 0)}")
        print(f"Main Directory: {local_stats.get('main_directory_files', 0)}")
        print(f"Organized Files: {local_stats.get('organized_files', 0)}")
        print(f"Backup Files: {local_stats.get('backup_files', 0)}")
        print(f"Total Size: {local_stats.get('total_size_mb', 0):.2f} MB")
        print(f"Storage Path: {local_stats.get('storage_path', 'N/A')}")
    
    # Configuration
    print("\n⚙️ CONFIGURATION")
    print("-"*15)
    config = stats['configuration']
    print(f"Chunk Size: {config.get('chunk_size', 0)}")
    print(f"Chunk Overlap: {config.get('chunk_overlap', 0)}")
    print(f"Top K Results: {config.get('top_k_results', 0)}")
    print(f"Similarity Threshold: {config.get('similarity_threshold', 0)}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main() 